import os
import shutil
import logging
import sys
import difflib
import re
from collections import defaultdict

# Local module imports
from asr import transcribe_audio
from audio_splitter import split_batch
import audio_utils as au
from ser import SpeechEmotionRecognizer

class DatasetProcessor:
    """封装整个数据集处理流程的类"""

    def __init__(self, config):
        """
        初始化处理器。
        :param config: 从 config.yaml 加载的配置字典。
        """
        self.config = config
        self.root_dir = os.path.abspath(config['root_dir'])
        
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"配置中指定的根目录不存在或不是目录: {self.root_dir}")

        # 从配置中提取参数
        self.recursive = self.config.get('recursive', False)
        self.use_lab_text_as_filename = self.config.get('use_lab_text_as_filename', False)
        
        split_merge_config = self.config.get('split_and_merge', {})
        self.output_to_split_folder = split_merge_config.get('output_to_split_folder', True)

        text_config = self.config.get('text', {})
        self.replace_map = text_config.get('replace_map', {})
        self.t2s_enabled = text_config.get('t2s', True)

        self.asr_config = self.config.get('asr', {})
        
        self.ser_config = self.config.get('ser', {})
        self.ser_enabled = self.ser_config.get('enable', False)
        self.ser_recognizer = None
        if self.ser_enabled:
            self.ser_recognizer = SpeechEmotionRecognizer(self.ser_config)

        output_config = self.config.get('output', {})
        self.jsonl_output_enabled = output_config.get('jsonl_output', False)

        self.folder_name = os.path.basename(self.root_dir.rstrip(os.sep))
        if not self.folder_name:
            raise ValueError(f"无法从根目录派生文件夹名称: {self.root_dir}")

        # 路径和日志记录器将在 setup 阶段初始化
        self.base_output_dir = None
        self.output_dir = None
        self.split_output_dir = None # 切分文件专属输出目录
        self.temp_dir = None
        self.auto_merge_dir = None
        self.auto_split_dir = None
        self.log_path = None
        # self.out_list_path = None # 将被替换为按情感分类的列表
        self.logger = None

    def _extract_asr_for_placeholder(self, asr_segments, prefix, suffix):
        """
        基于文本模糊匹配和索引映射的 {ASR} 占位符提取。
        """
        if not asr_segments:
            return None

        full_asr_text = "".join(seg["text"] for seg in asr_segments)

        # 1. 创建从规范化文本索引到原始文本索引的映射
        norm_to_full_map = []
        norm_char_idx = 0
        for i, char_full in enumerate(full_asr_text):
            # 只有非标点字符才会计入规范化文本的长度
            if char_full not in '，。！？、,.!?…“”‘’：;；（）()《》〈〉【】':
                norm_to_full_map.append(i) # 存储原始文本中的索引
                norm_char_idx += 1
        
        # 2. 规范化所有文本，用于模糊匹配
        def normalize(s: str) -> str:
            if not s:
                return ""
            # 移除所有标点符号和空白字符
            return re.sub(r'[\s，。！？、,.!?…“”‘’：;；（）()《》〈〉【】]', '', s)

        norm_full_asr_text = normalize(full_asr_text)
        norm_prefix = normalize(prefix)
        norm_suffix = normalize(suffix)

        if not norm_full_asr_text:
            return None

        MIN_MATCH_RATIO = 0.7 # 模糊匹配的最小匹配率阈值

        # 辅助函数：在规范化文本中查找匹配的起始和结束索引
        def find_match_indices_in_norm(target_norm, pattern_norm):
            if not pattern_norm:
                return None, None
            
            matcher = difflib.SequenceMatcher(None, target_norm, pattern_norm)
            match = matcher.find_longest_match(0, len(target_norm), 0, len(pattern_norm))
            
            if match.size > 0 and (match.size / len(pattern_norm)) >= MIN_MATCH_RATIO:
                return match.a, match.a + match.size # 返回在规范化文本中的起始和结束索引
            return None, None

        # 3. 在规范化文本上进行模糊匹配
        prefix_start_norm, prefix_end_norm = find_match_indices_in_norm(norm_full_asr_text, norm_prefix)
        suffix_start_norm, suffix_end_norm = find_match_indices_in_norm(norm_full_asr_text, norm_suffix)

        # 4. 将规范化文本的索引映射回原始文本
        start_idx_full = 0
        if prefix_end_norm is not None:
            # 确保索引在映射范围内
            if prefix_end_norm < len(norm_to_full_map):
                start_idx_full = norm_to_full_map[prefix_end_norm]
            else:
                # 如果前缀匹配到规范化文本的末尾或超出，则从原始文本的末尾开始
                start_idx_full = len(full_asr_text)

        end_idx_full = len(full_asr_text)
        if suffix_start_norm is not None:
            # 确保索引在映射范围内
            if suffix_start_norm < len(norm_to_full_map):
                end_idx_full = norm_to_full_map[suffix_start_norm]
            else:
                # 如果后缀匹配到规范化文本的末尾或超出，则到原始文本的末尾
                end_idx_full = len(full_asr_text)

        # 5. 确保索引有效且顺序正确
        if start_idx_full >= end_idx_full:
            return None

        # 6. 从原始文本中提取内容
        extracted_text = full_asr_text[start_idx_full:end_idx_full].strip()
        return extracted_text if extracted_text else None

    def _process_lab_with_asr(self, lab_path, wav_path):
        """
        支持 {ASR} 占位符的智能替换：仅替换占位符部分，保留上下文。
        """
        with open(lab_path, "r", encoding="utf-8") as f:
            original_lab = f.read().strip()

        temp_lab = original_lab
        for k, v in self.replace_map.items():
            temp_lab = temp_lab.replace(k, v)

        if "{ASR}" not in temp_lab:
            return temp_lab

        if self.logger:
            self.logger.info(f"检测到 {{ASR}} 占位符，对 {wav_path} 执行 ASR...")

        try:
            segments = transcribe_audio(wav_path, self.asr_config)
            
            if not segments:
                raise RuntimeError("ASR 未返回任何结果。")

            asr_full_text = "".join(seg["text"] for seg in segments).strip()
            asr_full_text = au.traditional_to_simplified(asr_full_text, self.t2s_enabled)
            if self.logger:
                self.logger.info(f"ASR 全文结果: {asr_full_text}")

            final_lab = temp_lab
            while "{ASR}" in final_lab:
                parts = final_lab.split("{ASR}", 1)
                prefix = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""

                extracted = self._extract_asr_for_placeholder(segments, prefix, suffix)

                if extracted is not None:
                    replacement = extracted
                    if self.logger:
                        self.logger.info(f"成功提取占位符内容: '{extracted}'")
                else:
                    replacement = "{ASR_FAILED}"
                    if self.logger:
                        self.logger.warning("无法从 ASR 结果中定位占位符内容，保留失败标记")

                final_lab = prefix + replacement + suffix

            return final_lab

        except Exception as e:
            error_msg = f"ASR 处理失败: {e}"
            if self.logger:
                self.logger.warning(error_msg)
            else:
                print(f"[WARN] {error_msg}", file=sys.stderr)
            return temp_lab

    def _setup_paths_and_logging(self):
        """设置所有输出路径并配置日志记录。"""
        self.base_output_dir = os.path.abspath("output")
        self.output_dir = os.path.join(self.base_output_dir, self.folder_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] 主要音频输出将被保存到: {self.output_dir}")

        if self.output_to_split_folder:
            self.split_output_dir = os.path.join(self.output_dir, "切分后")
            os.makedirs(self.split_output_dir, exist_ok=True)
            print(f"[INFO] 切分后的音频将被保存到: {self.split_output_dir}")
        else:
            self.split_output_dir = self.output_dir # 如果不单独输出，则指向主输出目录
        
        print(f"[INFO] 日志和列表文件将被保存到: {self.base_output_dir}")

        self.temp_dir = os.path.join(self.output_dir, "_temp_processing")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        self.auto_merge_dir = self.temp_dir
        self.auto_split_dir = self.temp_dir

        self.log_path = os.path.join(self.base_output_dir, f"{self.folder_name}.log")
        # self.out_list_path is now handled dynamically based on emotion
        # 将去重记录移动到角色文件夹下
        self.dedup_log_path = os.path.join(self.output_dir, "！去重记录.txt")

        self.logger = logging.getLogger("tts_preprocess")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self.logger.info("="*50)
        self.logger.info("开始新的处理流程")
        self.logger.info("配置信息:")
        self.logger.info(f"  - 输入目录: {self.root_dir}")
        self.logger.info(f"  - 输出目录: {self.output_dir}")
        self.logger.info(f"  - 切分文件输出到 '切分后' 文件夹: {self.output_to_split_folder}")
        self.logger.info(f"  - 使用标注文本命名: {self.use_lab_text_as_filename}")
        self.logger.info(f"  - 情感识别 (SER) 已 {'启用' if self.ser_enabled else '禁用'}")
        self.logger.info(f"  - 替换规则: {self.replace_map}")
        self.logger.info("="*50)

    def _find_audio_entries(self):
        """查找、配对、去重并读取所有音频条目及其元数据。"""
        import datetime
        stems = {}

        if self.recursive:
            self.logger.info(f"开始递归扫描目录: {self.root_dir}")
            for dirpath, _, filenames in os.walk(self.root_dir):
                for fname in filenames:
                    name, ext = os.path.splitext(fname)
                    ext_lower = ext.lower()
                    if ext_lower in (".wav", ".lab"):
                        # 使用相对路径和文件名作为唯一键，避免冲突
                        relative_path = os.path.relpath(os.path.join(dirpath, name), self.root_dir)
                        unique_stem_key = os.path.normpath(relative_path)
                        
                        if unique_stem_key not in stems:
                            stems[unique_stem_key] = {"dirpath": dirpath}
                        stems[unique_stem_key][ext_lower] = fname
        else:
            self.logger.info(f"开始扫描单个目录: {self.root_dir}")
            entries = os.listdir(self.root_dir)
            file_entries = [f for f in entries if os.path.isfile(os.path.join(self.root_dir, f))]
            for fname in file_entries:
                name, ext = os.path.splitext(fname)
                ext_lower = ext.lower()
                if ext_lower in (".wav", ".lab"):
                    if name not in stems:
                        stems[name] = {"dirpath": self.root_dir}
                    stems[name][ext_lower] = fname
        
        self.logger.info(f"找到 {len(stems)} 个带有 .wav 或 .lab 扩展名的唯一文件主干。")

        matched = []
        skipped_pairing = 0
        read_failures = []
        seen_texts = set()
        duplicate_log_lines = []
        skipped_rule_count = 0

        # 按文件名排序以确保去重行为是确定的
        stems_sorted = sorted(stems.items())

        for stem_key, extmap in stems_sorted:
            if ".wav" not in extmap or ".lab" not in extmap:
                skipped_pairing += 1
                continue

            dirpath = extmap["dirpath"]
            # 从 extmap 中获取原始文件名，因为 stem_key 可能包含路径
            wav_filename = extmap[".wav"]
            lab_filename = extmap[".lab"]
            
            wav_path = os.path.join(dirpath, wav_filename)
            lab_path = os.path.join(dirpath, lab_filename)

            try:
                lab_text = self._process_lab_with_asr(lab_path, wav_path)
            except Exception as e:
                read_failures.append(f"{lab_path} (处理 .lab): {e}")
                self.logger.warning(f"处理 .lab {lab_path} 失败: {e}")
                continue

            # {SKIP} rule check
            if "{SKIP}" in lab_text:
                self.logger.info(f"检测到 {{SKIP}} 规则，跳过文件: {os.path.basename(wav_path)}")
                skipped_rule_count += 1
                continue

            # 去重逻辑
            if lab_text in seen_texts:
                log_line = f"丢弃 (文本内容重复): {os.path.basename(wav_path)}"
                duplicate_log_lines.append(log_line)
                self.logger.info(f"发现重复文本，跳过文件: {os.path.basename(wav_path)}")
                continue
            else:
                seen_texts.add(lab_text)

            try:
                duration = au.get_wav_duration_seconds(wav_path)
            except Exception as e:
                read_failures.append(f"{wav_path} (获取时长): {e}")
                self.logger.warning(f"获取 {wav_path} 的时长失败: {e}")
                continue

            emotion = "neutral" # Default emotion
            if self.ser_enabled:
                self.logger.info(f"Running SER for {os.path.basename(wav_path)}")
                emotion = self.ser_recognizer.predict_emotion(wav_path)

            matched.append({
                "stem": stem_key, "wav_path": wav_path, "lab_path": lab_path,
                "lab_text": lab_text, "duration": duration, "emotion": emotion
            })

        if read_failures:
            print(f"[WARN] 文件读取期间遇到 {len(read_failures)} 个错误。详情请查看日志文件: {self.log_path}", file=sys.stderr)

        if duplicate_log_lines:
            try:
                with open(self.dedup_log_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n--- 去重记录 ({timestamp}) ---\n")
                    f.write("以下文件因 .lab 文本内容重复而被跳过处理：\n")
                    f.write("\n".join(duplicate_log_lines))
                    f.write("\n")
                print(f"[INFO] 发现并跳过了 {len(duplicate_log_lines)} 个重复文件。详情请查看: {self.dedup_log_path}")
                self.logger.info(f"追加 {len(duplicate_log_lines)} 条重复记录到 {self.dedup_log_path}")
            except Exception as e:
                print(f"[WARN] 写入去重日志文件失败: {e}", file=sys.stderr)
                self.logger.error(f"写入去重日志文件失败: {e}")

        total_skipped = skipped_pairing + len(read_failures) + len(duplicate_log_lines) + skipped_rule_count
        self.logger.info(f"匹配到 {len(matched)} 对 .wav 和 .lab 文件。")
        if total_skipped > 0:
            self.logger.warning(f"共跳过了 {total_skipped} 个文件（{skipped_pairing} 个缺少配对, {len(read_failures)} 个读取失败, {len(duplicate_log_lines)} 个内容重复, {skipped_rule_count} 个触发SKIP规则）。")

        return matched

    def _process_short_files(self, short_list):
        """处理短音频，按情感分组后进行自动合并，并根据新规则处理剩余的孤立文件。"""
        if not short_list:
            return [], []

        # 1. 按情感对短音频进行分组
        emotion_groups = defaultdict(list)
        for item in short_list:
            emotion_groups[item['emotion']].append(item)
        
        self.logger.info(f"短音频按情感分组: { {e: len(v) for e, v in emotion_groups.items()} }")
        print(f"[INFO] Short audios grouped by emotion: { {e: len(v) for e, v in emotion_groups.items()} }")

        processed_entries = []
        merge_map_lines = []
        leftover_items = []

        # 2. 分离可合并的组和剩余的组
        for emotion, items in emotion_groups.items():
            if len(items) > 1:
                # 这些组足够大，可以进行标准合并
                self.logger.info(f"Emotion group '{emotion}' has {len(items)} items and will be merged.")
                
                s = items[:]
                groups = []
                n = len(s)
                if n in [2, 3]:
                    groups = [s]
                elif n > 3:
                    r = n % 3
                    main_count = n - (4 if r == 1 else r)
                    idx = 0
                    while idx < main_count:
                        groups.append(s[idx:idx+3])
                        idx += 3
                    tail = s[main_count:]
                    if r == 1:
                        groups.append(tail[0:2])
                        groups.append(tail[2:4])
                    elif r == 2:
                        groups.append(tail)

                for group_idx, group in enumerate(groups, start=1):
                    wav_infos = []
                    for item in group:
                        try:
                            params, _ = au.read_wav_params_and_frames(item["wav_path"])
                            wav_infos.append({"params": params, "path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"]})
                        except Exception as e:
                            self.logger.warning(f"Reading wav for merge failed {item['wav_path']}: {e}")
                            processed_entries.append(item)
                    
                    merged_duration = sum(i['duration'] for i in wav_infos)
                    merge_name = f"{self.folder_name}_{emotion}_merge_{group_idx}_{au.format_hms_filename(merged_duration)}.wav"
                    merge_path = os.path.join(self.auto_merge_dir, merge_name)

                    try:
                        au.merge_wavs(merge_path, wav_infos, self.config)
                        merge_lab = "".join(i["lab_text"] for i in wav_infos)
                        processed_entries.append({"wav_abs": os.path.abspath(merge_path), "lab_text": merge_lab, "duration": merged_duration, "emotion": emotion})
                        
                        source_files_str = ", ".join([f"{os.path.basename(i['path'])} ({i['emotion']})" for i in wav_infos])
                        merge_map_lines.append(f"{os.path.basename(merge_path)} ({emotion}) <- [{source_files_str}]")
                    except Exception as e:
                        self.logger.error(f"Merging group {group_idx} for emotion '{emotion}' failed: {e}")
                        processed_entries.extend(group)

            elif len(items) == 1:
                item = items[0]
                warning_msg = f"无法找到与 {os.path.basename(item['wav_path'])} ({item['emotion']}) 具有相同情感的音频进行合并，已暂存等待跨情感合并。"
                self.logger.warning(warning_msg)
                merge_map_lines.append(f"[警告] {warning_msg}")
                leftover_items.append(item)

        # 3. 合并所有剩余的孤立文件
        if len(leftover_items) > 1:
            self.logger.warning(f"正在合并 {len(leftover_items)} 个来自不同情感类别的孤立音频。")
            print(f"[WARN] Merging {len(leftover_items)} leftover audios from different emotion groups.")
            
            s = leftover_items[:]
            groups = []
            n = len(s)
            if n in [2, 3]:
                groups = [s]
            elif n > 3:
                r = n % 3
                main_count = n - (4 if r == 1 else r)
                idx = 0
                while idx < main_count:
                    groups.append(s[idx:idx+3])
                    idx += 3
                tail = s[main_count:]
                if r == 1:
                    groups.append(tail[0:2])
                    groups.append(tail[2:4])
                elif r == 2:
                    groups.append(tail)

            for group_idx, group in enumerate(groups, start=1):
                wav_infos = []
                for item in group:
                    try:
                        params, _ = au.read_wav_params_and_frames(item["wav_path"])
                        wav_infos.append({"params": params, "path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"]})
                    except Exception as e:
                        self.logger.warning(f"Reading leftover wav for merge failed {item['wav_path']}: {e}")
                        processed_entries.append(item)

                if not wav_infos: continue

                merged_duration = sum(i['duration'] for i in wav_infos)
                # 新规则：合并后的情感 = 第一个音频的情感
                result_emotion = wav_infos[0]['emotion']
                
                merge_name = f"{self.folder_name}_cross-emotion_merge_{group_idx}_{au.format_hms_filename(merged_duration)}.wav"
                merge_path = os.path.join(self.auto_merge_dir, merge_name)

                try:
                    au.merge_wavs(merge_path, wav_infos, self.config)
                    merge_lab = "".join(i["lab_text"] for i in wav_infos)
                    processed_entries.append({"wav_abs": os.path.abspath(merge_path), "lab_text": merge_lab, "duration": merged_duration, "emotion": result_emotion})
                    
                    source_files_str = ", ".join([f"{os.path.basename(i['path'])} ({i['emotion']})" for i in wav_infos])
                    log_msg = f"{os.path.basename(merge_path)} ({result_emotion}) <- [{source_files_str}] (跨情感合并，选用第一个音频的情感)"
                    merge_map_lines.append(log_msg)
                    self.logger.info(log_msg)

                except Exception as e:
                    self.logger.error(f"Merging leftover group {group_idx} failed: {e}")
                    processed_entries.extend(group)
        
        elif len(leftover_items) == 1:
            # 如果最后只剩一个，无法合并，直接加入最终列表，并修正key
            item = leftover_items[0]
            processed_entries.append({
                "wav_abs": os.path.abspath(item["wav_path"]),
                "lab_text": item["lab_text"],
                "duration": item["duration"],
                "emotion": item["emotion"]
            })

        return processed_entries, merge_map_lines

    def _process_long_files(self, long_list):
        """处理长音频，进行自动切分，并返回切分后的文件信息。"""
        if not long_list:
            return [], []

        print(f"[INFO] 准备切分 {len(long_list)} 个长音频文件。")
        self.logger.info(f"准备对 {len(long_list)} 个长音频进行批量切分。")

        pairs = [(m["wav_path"], m["lab_text"]) for m in long_list]
        unsplit_entries = []
        split_entries = []
        
        try:
            batch_results = split_batch(pairs, self.auto_split_dir, self.config, logger_arg=self.logger)
        except Exception as e:
            self.logger.error(f"批量切分函数'split_batch'执行失败: {e}")
            # 如果批量切分失败，所有长音频都作为未切分处理，并保留其情感
            unsplit_entries.extend(long_list)
            return unsplit_entries, split_entries

        split_failures = []
        for m in long_list:
            wavp = m["wav_path"]
            original_emotion = m["emotion"] # 获取原始情感
            res = batch_results.get(wavp, {"paths": None, "error": "no result"})
            split_wav_paths = res.get("paths")
            
            if split_wav_paths:
                self.logger.info(f"切分成功: {wavp}")
                for split_wav_path in split_wav_paths:
                    try:
                        split_lab_path = os.path.splitext(split_wav_path)[0] + ".lab"
                        lab_text = au.read_lab(split_lab_path).strip()
                        duration = au.get_wav_duration_seconds(split_wav_path)
                        
                        # 切分后的文件继承原始情感
                        split_entries.append({
                            "wav_abs": os.path.abspath(split_wav_path),
                            "lab_text": lab_text,
                            "duration": duration,
                            "is_split": True,
                            "emotion": original_emotion 
                        })
                    except Exception as e:
                        self.logger.error(f"处理切分后的文件 {split_wav_path} 失败: {e}")
            else:
                split_failures.append(os.path.basename(wavp))
                # 切分失败的文件保留其原始信息，但要确保 key 是 'wav_abs'
                unsplit_entries.append({
                    "wav_abs": os.path.abspath(m["wav_path"]),
                    "lab_text": m["lab_text"],
                    "duration": m["duration"],
                    "emotion": original_emotion
                })
        
        if split_failures:
            print(f"[WARN] {len(split_failures)} 个长音频文件自动切分失败。", file=sys.stderr)

        return unsplit_entries, split_entries

    def _finalize_output(self, final_entries, merge_map_lines, initial_count):
        """最终确定输出文件：重命名、复制并写入所有摘要和列表文件。"""
        self.logger.info(f"开始最后的文件整理与输出，共 {len(final_entries)} 个最终音频。")
        
        final_output_entries_by_emotion = defaultdict(list)
        processed_filenames_by_emotion = defaultdict(set)

        for entry in final_entries:
            source_path = entry["wav_abs"]
            lab_text = entry["lab_text"]
            is_split = entry.get("is_split", False)
            emotion = entry.get("emotion", "neutral")

            # 根据文件类型和情感确定输出目录
            if self.ser_enabled:
                target_dir = os.path.join(self.output_dir, emotion)
            else:
                target_dir = self.output_dir

            if is_split and self.output_to_split_folder:
                # 如果启用了SER，切分文件夹也应该在情感目录下
                target_dir = os.path.join(target_dir, "切分后")

            os.makedirs(target_dir, exist_ok=True)

            if self.use_lab_text_as_filename:
                target_basename = au.sanitize_filename(lab_text) + ".wav"
            else:
                source_filename = os.path.basename(source_path)
                if is_split:
                    target_basename = source_filename
                elif source_path.startswith(os.path.abspath(self.auto_merge_dir)):
                    # 使用更具描述性的合并文件名
                    target_basename = source_filename
                else:
                    target_basename = source_filename

            # 确保在各自的情感目录中文件名唯一
            temp_basename, counter = target_basename, 1
            while temp_basename in processed_filenames_by_emotion[emotion]:
                base, ext = os.path.splitext(target_basename)
                temp_basename = f"{base}_{counter}{ext}"
                counter += 1
            target_basename = temp_basename
            processed_filenames_by_emotion[emotion].add(target_basename)

            target_path = os.path.join(target_dir, target_basename)
            try:
                if source_path.startswith(os.path.abspath(self.temp_dir)):
                    shutil.move(source_path, target_path)
                else:
                    shutil.copy(source_path, target_path)
                
                entry_copy = entry.copy()
                entry_copy["wav_abs"] = os.path.abspath(target_path)
                final_output_entries_by_emotion[emotion].append(entry_copy)

            except Exception as e:
                self.logger.error(f"复制或移动 {source_path} 到 {target_path} 失败: {e}")

        if merge_map_lines:
            with open(os.path.join(self.output_dir, "！合并记录.txt"), "a", encoding="utf-8") as mf:
                mf.write("\n".join(merge_map_lines) + "\n")

        split_log_path = os.path.join(self.temp_dir, "切分记录.txt")
        if os.path.exists(split_log_path):
            target_log_path = os.path.join(self.output_dir, "！切分记录.txt")
            shutil.move(split_log_path, target_log_path)

        total_duration_seconds = 0
        self.logger.info("--- 最终结果统计 ---")
        print("\n--- Final Results ---")

        # 如果SER禁用，但仍有条目，则它们都将被视为 "neutral"
        if not self.ser_enabled and any(final_output_entries_by_emotion.values()):
             if "neutral" not in final_output_entries_by_emotion:
                 # This can happen if all files were processed without SER and ended up in other categories
                 # For summary purposes, we can lump them under neutral.
                 all_entries = []
                 for emotion_group in final_output_entries_by_emotion.values():
                     all_entries.extend(emotion_group)
                 final_output_entries_by_emotion.clear()
                 final_output_entries_by_emotion["neutral"] = all_entries

        for emotion, entries in final_output_entries_by_emotion.items():
            emotion_duration = sum(e['duration'] for e in entries)
            total_duration_seconds += emotion_duration
            summary_text = f"情感 '{emotion}': {len(entries)} 个文件, 总时长: {au.format_hms(emotion_duration)}"
            print(summary_text)
            self.logger.info(summary_text)

            # 为每个情感创建独立的时长文件
            formatted_duration = au.format_hms_filename(emotion_duration)
            duration_filename = f"！{formatted_duration}_{emotion}.txt"
            duration_filepath = os.path.join(self.output_dir, duration_filename)
            with open(duration_filepath, "w", encoding="utf-8") as f:
                pass
            self.logger.info(f"已创建情感时长文件: {duration_filepath}")

            # 为每个情感生成独立的 .list 文件
            if self.ser_enabled:
                list_path = os.path.join(self.base_output_dir, f"{self.folder_name}_{emotion}.list")
            else:
                # 如果SER禁用，则使用旧的单一文件格式
                list_path = os.path.join(self.base_output_dir, f"{self.folder_name}.list")

            with open(list_path, "w", encoding="utf-8") as ol:
                for ent in entries:
                    ol.write(f"{ent['wav_abs']}|{self.folder_name}|ZH|{ent['lab_text']}\n")
            self.logger.info(f"已生成列表文件: {list_path}")
            print(f"Generated list file: {list_path}")

            # 如果启用了 JSONL 输出，则生成对应的 .jsonl 文件
            if self.jsonl_output_enabled:
                import json
                jsonl_path = os.path.splitext(list_path)[0] + ".jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
                    for ent in entries:
                        json_record = {
                            "id": os.path.basename(ent['wav_abs']),
                            "audio": ent['wav_abs'],
                            "text": ent['lab_text'],
                            "speaker": self.folder_name
                        }
                        f_jsonl.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                self.logger.info(f"已生成 JSONL 文件: {jsonl_path}")
                print(f"Generated JSONL file: {jsonl_path}")

        overall_summary = f"最终音频总时长: {au.format_hms(total_duration_seconds)}"
        print(overall_summary)
        self.logger.info(overall_summary)

    def run(self):
        """执行完整的数据集处理流程。"""
        self._setup_paths_and_logging()
        all_entries = self._find_audio_entries()

        thresholds = self.config.get('split_and_merge', {'short_threshold': 2.0, 'long_threshold': 15.0})
        short_threshold = thresholds.get('short_threshold', 2.0)
        long_threshold = thresholds.get('long_threshold', 15.0)

        short_list = [m for m in all_entries if m["duration"] <= short_threshold]
        normal_list = [m for m in all_entries if short_threshold < m["duration"] < long_threshold]
        long_list = [m for m in all_entries if m["duration"] >= long_threshold]
        
        final_entries_before_copy = []
        # 添加正常时长的文件, 并确保 key 为 'wav_abs'
        final_entries_before_copy.extend([
            {
                "wav_abs": os.path.abspath(m["wav_path"]),
                "lab_text": m["lab_text"],
                "duration": m["duration"],
                "emotion": m["emotion"]
            } for m in normal_list
        ])
        
        # 处理长音频，同时接收未切分成功和已切分成功的文件
        unsplit_long_entries, split_entries = self._process_long_files(long_list)
        
        # 对未切分成功的长音频，同样确保 key 为 'wav_abs'
        for m in unsplit_long_entries:
            final_entries_before_copy.append({
                "wav_abs": os.path.abspath(m["wav_path"]),
                "lab_text": m["lab_text"],
                "duration": m["duration"],
                "emotion": m["emotion"]
            })

        final_entries_before_copy.extend(split_entries)
        
        # 处理短音频
        processed_short_entries, merge_map_lines = self._process_short_files(short_list)
        final_entries_before_copy.extend(processed_short_entries)
        
        # 最终化输出
        self._finalize_output(final_entries_before_copy, merge_map_lines, len(all_entries))
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理处理过程中生成的临时目录。"""
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"已清理临时目录: {self.temp_dir}")
            except Exception as e:
                self.logger.error(f"清理临时目录 {self.temp_dir} 失败: {e}")