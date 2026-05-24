# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Optional, Dict, Any, Tuple, Set
from PIL import Image, ImageOps
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from functools import lru_cache
import hashlib
import math
import re

from .base import BaseOCRTool
from .utils import (
    trajectory_to_image,
    text_to_adaptive_image_compact,
    apply_compact_mode,
    get_font_metrics,
    _get_cached_font,
    preprocess_trajectory_contexts,
    wrap_text_fast,
    wrap_text_precise,
    COMPACT_NEWLINE_SYMBOL
)
from .trust_policy import (
    GoalSlots,
    _line_summary_fact_keys,
    build_compact_trust_context,
    build_trust_policy_text_summary,
    MemorySkillFeedback,
    PreparedTrustContext,
    SegmentTrustMetadata,
    TrustCalibratedRenderPolicy,
    build_trust_segments_from_lines,
    build_query_conditioned_segments_from_lines,
    collect_trust_policy_monitor,
    _phase_aware_min_compaction_lines,
    prepare_trust_context,
)


class OCRMetadataArray(np.ndarray):
    """NumPy array wrapper that allows attaching render metadata."""


class SegmentCache:
    """
    Segment-level cache for AgentOCR.
    
    This cache stores rendered segment images keyed by segment content hash.
    It enables efficient reuse of rendered segments across different history states,
    supporting both recurring boilerplate and repeated tool outputs.
    
    Key features:
    - Each unique segment is rendered at most once
    - Segments can be reused whenever they match cached content
    - Supports sliding windows and non-contiguous history matching
    
    Cache structure:
        {content_hash: np.ndarray (rendered segment image)}
    """
    
    def __init__(self):
        """Initialize an empty segment cache."""
        self._cache: Dict[int, np.ndarray] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_lookups': 0,
            'segments_cached': 0,
        }
    
    def get_key(self, segment_text: str, cache_key_text: Optional[str] = None) -> int:
        """
        Generate a fast content key for a segment.
        
        Args:
            segment_text: The normalized segment text
            
        Returns:
            Hash key for the segment
        """
        key_text = cache_key_text if cache_key_text is not None else segment_text
        return hash(str(key_text).strip())
    
    def lookup(self, segment_text: str, cache_key_text: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Look up a segment in the cache.
        
        Args:
            segment_text: The segment text to look up
            
        Returns:
            Rendered segment image if found, None otherwise
        """
        self._stats['total_lookups'] += 1
        key = self.get_key(segment_text, cache_key_text=cache_key_text)
        
        if key in self._cache:
            self._stats['hits'] += 1
            return self._cache[key]
        else:
            self._stats['misses'] += 1
            return None
    
    def insert(
        self,
        segment_text: str,
        rendered_image: np.ndarray,
        cache_key_text: Optional[str] = None,
    ) -> None:
        """
        Insert a rendered segment into the cache.
        
        Args:
            segment_text: The segment text (used as key)
            rendered_image: The rendered image as numpy array
        """
        key = self.get_key(segment_text, cache_key_text=cache_key_text)
        if key not in self._cache:
            self._cache[key] = rendered_image.copy()
            self._stats['segments_cached'] += 1
    
    def contains(self, segment_text: str, cache_key_text: Optional[str] = None) -> bool:
        """Check if a segment is in the cache without updating stats."""
        key = self.get_key(segment_text, cache_key_text=cache_key_text)
        return key in self._cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats['total_lookups']
        hits = self._stats['hits']
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            'total_lookups': total,
            'hits': hits,
            'misses': self._stats['misses'],
            'hit_rate': f'{hit_rate:.1f}%',
            'segments_cached': self._stats['segments_cached'],
            'cache_size_mb': sum(img.nbytes for img in self._cache.values()) / (1024 * 1024),
        }
    
    def clear(self) -> None:
        """Clear the cache and reset statistics."""
        self._cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_lookups': 0,
            'segments_cached': 0,
        }
    
    def __len__(self) -> int:
        """Return the number of cached segments."""
        return len(self._cache)


def split_into_segments(history: str) -> List[str]:
    """
    Split a history string into segments.
    
    This is the Split(h) operation described in the paper.
    Each segment is a line of text split by newlines.
    
    Args:
        history: The full history string
        
    Returns:
        List of text segments (non-empty lines)
    """
    if not history:
        return []
    
    # Split by newlines and filter out empty lines
    # Each non-empty line becomes a segment
    segments = [line.strip() for line in history.split('\n') if line.strip()]
    return segments


class OCRTool(BaseOCRTool):
    """
    OCR Tool for converting trajectory history records (text) into images.
    
    This tool is designed to be:
    - Highly flexible: Supports various trajectory formats and configurations
    - Decoupled: Works independently of the main pipeline
    - Easy to integrate: Minimal modifications needed to environment managers
    - Optimized for sliding windows: Segment-based caching supports non-contiguous history
    
    Caching Strategy (Segment-Based):
        - Instead of caching only full prefixes, we cache individual segments (lines split by \n)
        - Segments are split by newlines to match memory structure exactly
        - Each segment has its own content hash and height range in master image
        - Supports sliding window: Can match and reuse segments from any position
        - Format-agnostic: No dependency on specific patterns like "Observation X:"
        - Example: If context changes from "line 1-5" to "line 3-7", lines 3-5 are reused
    
    Master Image Structure:
        - master_img: Single concatenated image containing all cached segments
        - segments: List of segment metadata (content_hash, step, start_h, end_h, text)
        - indices: Dict for backward compatibility (full context hash -> position)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        font_size: Optional[int] = 10,
        padding: int = 10,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        font_path: Optional[str] = None,
        min_width: int = 28,
        max_width: int = 1024,
        min_height: int = 28,
        max_height: int = 1024,
        max_workers: Optional[int] = None,
        use_parallel: bool = True,
        use_precise: bool = True,
        fast_mode: bool = True,
        enable_cache: bool = True,
        compact_mode: bool = False,
        compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
        highlight_configs: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the OCRTool with ultra-optimized settings for maximum text coverage
        and minimum resolution while maintaining clarity.
        
        Args:
            enabled: Whether the tool is enabled (can be toggled at runtime)
            font_size: Font size for text rendering
            padding: Padding around text in pixels
            bg_color: Background color as RGB tuple
            text_color: Text color as RGB tuple
            font_path: Path to custom font file
            min_width: Minimum image width in pixels
            max_width: Maximum image width in pixels
            min_height: Minimum image height in pixels
            max_height: Maximum image height in pixels
            max_workers: Maximum number of parallel workers (None for auto)
            use_parallel: Whether to use parallel processing for batch conversion
            use_precise: Use precise font measurements for optimal packing (recommended)
            fast_mode: Use fast mode (fixed width) for real-time performance (default True)
            enable_cache: Enable LRU caching of rendered images for speedup (default True)
            compact_mode: Enable compact mode (replace newlines with symbols)
            compact_symbol: Symbol to use for newline replacement in compact mode
            highlight_configs: List of dicts specifying text contexts to highlight with colors.
                              To highlight compact_symbol, include it in highlight_configs.
                              Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                       {"context": "⏎", "color": [128, 128, 128]}]
            **kwargs: Additional parameters passed to trajectory_to_image
        """
        self.enabled = enabled
        self.font_size = font_size
        self.padding = padding
        self.bg_color = tuple(bg_color)
        self.text_color = tuple(text_color)
        self.font_path = font_path
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.max_workers = max_workers if max_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        self.use_parallel = use_parallel
        self.use_precise = use_precise
        self.fast_mode = fast_mode
        self.enable_cache = enable_cache
        self.compact_mode = compact_mode
        self.compact_symbol = compact_symbol
        self.highlight_configs = highlight_configs
        self.kwargs = kwargs
        # Initialize folder for saving trajectory images
        self.trajectory_images_dir = os.path.join(os.getcwd(), "logs/trajectory_images")
        os.makedirs(self.trajectory_images_dir, exist_ok=True)
        self.image_save_counter = 0
        # Incremental rendering: use master image + height indices to save memory
        # Format: {env_idx: {'master_img': np.ndarray, 'indices': {step_range_hash: (start, end)}}}
        self._master_images = {} if enable_cache else None
        # Cache statistics
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        # Track last printed batch number for cache stats
        self._last_printed_cache_batch = 0
        
        # Segment-level cache for efficient rendering (paper: AgentOCR segment caching)
        # Each environment has its own segment cache: {env_idx: SegmentCache}
        self._segment_caches: Dict[int, SegmentCache] = {} if enable_cache else None
        # Segment cache statistics (aggregated across all environments)
        self._segment_cache_stats = {
            'total_segments': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'segments_rendered': 0,
            'segments_reused': 0,
        }
        # Compact mode cache: stores incomplete line text for each environment
        # Format: {env_idx: {'incomplete_text': str, 'complete_lines_img': np.ndarray, 'complete_lines_count': int}}
        self._compact_cache = {} if enable_cache and compact_mode else None
        # Compact mode cache statistics
        self._compact_cache_stats = {
            'total': 0,              # Total render requests
            'full_hits': 0,          # Exact context hash matches (no re-render needed)
            'partial_hits': 0,       # Reused complete lines from cache
            'misses': 0,             # No cache to reuse (first render or context changed)
            'no_complete_lines': 0,  # Content too short to have complete lines to cache
            'cached_lines_reused': 0,  # Total number of cached lines reused
            'lines_rendered': 0,     # Total number of lines actually rendered
        }
        self._last_trust_policy_diagnostics: List[Dict[str, Any]] = []
        self._last_trust_policy_processed_contexts: List[str] = []
        self._last_trust_policy_prompt_summaries: List[str] = []
        self._last_applied_compression_factors: List[float] = []
        self._last_qwen3_history_layout_diagnostics: List[Dict[str, Any]] = []
    
    def convert(
        self,
        trajectory_text: Union[str, List[str]],
        **override_kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Convert trajectory text to image(s).
        
        Args:
            trajectory_text: Single trajectory text string or list of trajectory texts
            **override_kwargs: Parameters to override default configuration
        
        Returns:
            PIL Image object or list of PIL Image objects
        """
        if not self.is_enabled():
            return None if isinstance(trajectory_text, str) else [None] * len(trajectory_text)
        
        # Merge default config with override parameters
        config = self._get_config(**override_kwargs)
        
        # Handle both single string and list of strings
        if isinstance(trajectory_text, str):
            return self._convert_single(trajectory_text, config)
        else:
            return [self._convert_single(text, config) for text in trajectory_text]
    
    def convert_batch(
        self,
        trajectory_texts: List[str],
        **override_kwargs
    ) -> List[Image.Image]:
        """
        Convert a batch of trajectory texts to images with optional parallel processing.
        
        Args:
            trajectory_texts: List of trajectory text strings
            **override_kwargs: Parameters to override default configuration
        
        Returns:
            List of PIL Image objects
        """
        if not self.is_enabled():
            return [None] * len(trajectory_texts)
        
        if not trajectory_texts:
            return []
        
        # Merge default config with override parameters
        config = self._get_config(**override_kwargs)
        
        # Use parallel processing for batches larger than 1 if enabled
        if self.use_parallel and len(trajectory_texts) > 1:
            return self._convert_batch_parallel(trajectory_texts, config)
        else:
            return [self._convert_single(text, config) for text in trajectory_texts]
    
    def _convert_batch_parallel(
        self,
        trajectory_texts: List[str],
        config: Dict[str, Any]
    ) -> List[Image.Image]:
        """
        Convert a batch of trajectory texts to images using parallel processing.
        
        Args:
            trajectory_texts: List of trajectory text strings
            config: Configuration dictionary
        
        Returns:
            List of PIL Image objects (in the same order as input)
        """
        results = [None] * len(trajectory_texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._convert_single, text, config): idx
                for idx, text in enumerate(trajectory_texts)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Fallback to blank image on error
                    results[idx] = Image.new(
                        'RGB',
                        (self.min_width, self.min_height),
                        self.bg_color
                    )
        
        return results
    
    def _convert_single(
        self,
        trajectory_text: str,
        config: Dict[str, Any]
    ) -> Image.Image:
        """
        Convert a single trajectory text to an image with optimized packing.
        
        Args:
            trajectory_text: Trajectory text string
            config: Configuration dictionary
        
        Returns:
            PIL Image object with optimally packed text
        """
        trajectory_text = trajectory_text.strip()
        if not trajectory_text:
            # Return a blank image if trajectory is empty
            return Image.new(
                'RGB',
                (self.min_width, self.min_height),
                self.bg_color
            )
        
        
        # Render image
        img = trajectory_to_image(
            trajectory_text,
            font_size=config['font_size'],
            padding=config['padding'],
            bg_color=config['bg_color'],
            text_color=config['text_color'],
            font_path=config['font_path'],
            min_width=config['min_width'],
            max_width=config['max_width'],
            min_height=config['min_height'],
            max_height=config['max_height'],
            use_precise=config['use_precise'],
            fast_mode=config['fast_mode'],
            compact_mode=config['compact_mode'],
            compact_symbol=config['compact_symbol'],
            highlight_configs=config['highlight_configs'],
            **config['extra_kwargs']
        )
        
        
        return img

    def _render_lines(
        self,
        lines: List[str],
        **override_kwargs
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Render a list of lines into a stacked image and return per-line height ranges.
        """
        if not lines:
            blank = self._get_blank_array(**override_kwargs)
            return blank, [(0, blank.shape[0])]
        
        # Render without any padding; padding will be added later in the pipeline.
        render_kwargs = {**override_kwargs, 'padding': 0, 'min_height': 0}
        images = self.convert_batch(lines, **render_kwargs)
        arrays = []
        ranges: List[Tuple[int, int]] = []
        current_h = 0
        
        for img in images:
            arr = np.array(img) if img is not None else self._get_blank_array(**override_kwargs)
            start_h = current_h
            current_h += arr.shape[0]
            ranges.append((start_h, current_h))
            arrays.append(arr)
        
        stacked = arrays[0] if len(arrays) == 1 else np.vstack(arrays)
        return stacked, ranges
    
    def _get_cache_key(self, text: str, config: Dict[str, Any]) -> str:
        """Generate a cache key for a text and config combination."""
        # Use hash for efficient key generation
        config_str = f"{config['font_size']}_{config['padding']}"
        config_str += f"_{config['min_width']}_{config['max_width']}_{config['use_precise']}_{config['fast_mode']}"
        key = f"{hash(text)}_{config_str}"
        return key
    
    def _get_config(self, **override_kwargs) -> Dict[str, Any]:
        """
        Get configuration dictionary, merging defaults with overrides.
        
        Args:
            **override_kwargs: Parameters to override
        
        Returns:
            Configuration dictionary
        """
        # Extract extra kwargs that are not direct parameters
        extra_kwargs = {}
        direct_params = {
            'font_size', 'padding',
            'bg_color', 'text_color', 'font_path', 'min_width', 'max_width',
            'min_height', 'max_height', 'use_precise', 'fast_mode',
            'compact_mode', 'compact_symbol', 'highlight_configs'
        }
        
        for key, value in override_kwargs.items():
            if key not in direct_params:
                extra_kwargs[key] = value
        
        # Merge with instance kwargs
        extra_kwargs = {**self.kwargs, **extra_kwargs}
        
        return {
            'font_size': override_kwargs.get('font_size', self.font_size),
            'padding': override_kwargs.get('padding', self.padding),
            'bg_color': override_kwargs.get('bg_color', self.bg_color),
            'text_color': override_kwargs.get('text_color', self.text_color),
            'font_path': override_kwargs.get('font_path', self.font_path),
            'min_width': override_kwargs.get('min_width', self.min_width),
            'max_width': override_kwargs.get('max_width', self.max_width),
            'min_height': override_kwargs.get('min_height', self.min_height),
            'max_height': override_kwargs.get('max_height', self.max_height),
            'use_precise': override_kwargs.get('use_precise', self.use_precise),
            'fast_mode': override_kwargs.get('fast_mode', self.fast_mode),
            'compact_mode': override_kwargs.get('compact_mode', self.compact_mode),
            'compact_symbol': override_kwargs.get('compact_symbol', self.compact_symbol),
            'highlight_configs': override_kwargs.get('highlight_configs', self.highlight_configs),
            'extra_kwargs': extra_kwargs
        }
    
    def is_enabled(self) -> bool:
        """
        Check if the OCR tool is enabled and ready to use.
        
        Returns:
            True if the tool is enabled, False otherwise
        """
        return self.enabled
    
    def enable(self):
        """Enable the OCR tool."""
        self.enabled = True
    
    def disable(self):
        """Disable the OCR tool."""
        self.enabled = False
    
    def enable_compact_mode(self):
        """
        Enable compact mode (replace newlines with colored symbols).
        Initializes compact cache if not already present.
        """
        self.compact_mode = True
        if self._compact_cache is None and self.enable_cache:
            self._compact_cache = {}
    
    def disable_compact_mode(self):
        """
        Disable compact mode (use normal newline rendering).
        Clears compact cache to free memory.
        """
        self.compact_mode = False
        if self._compact_cache is not None:
            self._compact_cache.clear()
    
    def is_compact_mode(self) -> bool:
        """Check if compact mode is enabled."""
        return self.compact_mode
    
    def set_compact_symbol(self, symbol: str):
        """
        Set the symbol used for newline replacement in compact mode.
        
        Args:
            symbol: The symbol to use (e.g., '⏎', '↵', '¶')
        
        Note: To set the color, add the symbol to highlight_configs.
        """
        self.compact_symbol = symbol
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
    
    def clear_segment_cache(self, env_idx: Optional[int] = None) -> None:
        """
        Clear segment cache for a specific environment or all environments.
        
        Args:
            env_idx: Environment index to clear. If None, clears all caches.
        """
        if self._segment_caches is None:
            return
        
        if env_idx is not None:
            if env_idx in self._segment_caches:
                self._segment_caches[env_idx].clear()
        else:
            for cache in self._segment_caches.values():
                cache.clear()
            self._segment_caches.clear()
    
    def get_segment_cache(self, env_idx: int) -> Optional[SegmentCache]:
        """
        Get the segment cache for a specific environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            SegmentCache for the environment, or None if not initialized
        """
        if self._segment_caches is None:
            return None
        return self._segment_caches.get(env_idx)
    
    def reset(self):
        """
        Reset the OCR tool state, clearing all caches and statistics.
        This is useful when starting a new episode or batch of episodes.
        """
        # Clear master images cache
        if self._master_images is not None:
            self._master_images.clear()
        
        # Clear compact mode cache
        if self._compact_cache is not None:
            self._compact_cache.clear()
        
        # Clear segment caches for all environments
        if self._segment_caches is not None:
            self._segment_caches.clear()
        
        # Reset cache statistics
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        
        # Reset segment cache statistics
        self._segment_cache_stats = {
            'total_segments': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'segments_rendered': 0,
            'segments_reused': 0,
        }
        
        # Reset compact mode cache statistics
        self._compact_cache_stats = {
            'total': 0,
            'full_hits': 0,
            'partial_hits': 0,
            'misses': 0,
            'no_complete_lines': 0,
            'cached_lines_reused': 0,
            'lines_rendered': 0,
        }
        self._last_trust_policy_diagnostics = []
        self._last_trust_policy_processed_contexts = []
        self._last_trust_policy_prompt_summaries = []
        self._last_applied_compression_factors = []
        self._last_qwen3_history_layout_diagnostics = []

    def get_last_trust_policy_diagnostics(self) -> List[Dict[str, Any]]:
        return list(self._last_trust_policy_diagnostics)

    def get_last_trust_policy_processed_contexts(self) -> List[str]:
        return list(self._last_trust_policy_processed_contexts)

    def get_last_trust_policy_prompt_summaries(self) -> List[str]:
        return list(self._last_trust_policy_prompt_summaries)

    def get_last_qwen3_history_layout_diagnostics(self) -> List[Dict[str, Any]]:
        return list(self._last_qwen3_history_layout_diagnostics)

    def get_last_applied_compression_factors(self) -> List[float]:
        return list(self._last_applied_compression_factors)

    def _compute_trust_policy_mm_floor_metadata(self, diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = diagnostics or {}
        if float(metrics.get("trust_policy/render_compacted", 0.0)) <= 0.0:
            return {}

        raw_line_count = max(1.0, float(metrics.get("trust_policy/raw_line_count", 0.0)))
        rendered_line_count = max(1.0, float(metrics.get("trust_policy/rendered_line_count", raw_line_count)))
        line_keep_ratio = max(0.0, min(1.0, rendered_line_count / raw_line_count))
        raw_char_count = max(1.0, float(metrics.get("trust_policy/raw_char_count", 0.0)))
        rendered_char_count = max(1.0, float(metrics.get("trust_policy/rendered_char_count", raw_char_count)))
        char_keep_ratio = max(0.0, min(1.0, rendered_char_count / raw_char_count))

        try:
            floor_min = int(os.environ.get("AGENTOCR_QWEN3_TRUST_MM_MIN_VISUAL_TOKENS", "144"))
        except Exception:
            floor_min = 144
        try:
            floor_max = int(os.environ.get("AGENTOCR_QWEN3_TRUST_MM_MAX_VISUAL_TOKENS", "224"))
        except Exception:
            floor_max = 224
        floor_min = max(64, floor_min)
        floor_max = max(floor_min, floor_max)

        ratio_based_floor = int(round(floor_min + (floor_max - floor_min) * line_keep_ratio))
        ratio_based_floor = max(floor_min, min(floor_max, ratio_based_floor))

        try:
            size_cap_base = int(os.environ.get("AGENTOCR_QWEN3_TRUST_MM_SIZE_CAP_BASE_VISUAL_TOKENS", "64"))
        except Exception:
            size_cap_base = 64
        try:
            char_step = int(os.environ.get("AGENTOCR_QWEN3_TRUST_MM_SIZE_CAP_CHARS_PER_STEP", "96"))
        except Exception:
            char_step = 96
        try:
            line_step = int(os.environ.get("AGENTOCR_QWEN3_TRUST_MM_SIZE_CAP_LINES_PER_STEP", "1"))
        except Exception:
            line_step = 1
        try:
            token_step = int(os.environ.get("AGENTOCR_QWEN3_TRUST_MM_SIZE_CAP_TOKEN_STEP", "8"))
        except Exception:
            token_step = 8

        size_cap_base = max(64, min(floor_max, size_cap_base))
        char_step = max(1, char_step)
        line_step = max(1, line_step)
        token_step = max(1, token_step)
        char_steps = max(1, int(math.ceil(rendered_char_count / float(char_step))))
        line_steps = max(1, int(math.ceil(rendered_line_count / float(line_step))))
        char_based_cap = size_cap_base + token_step * char_steps
        line_based_cap = size_cap_base + token_step * line_steps
        size_based_cap = max(size_cap_base, min(floor_max, max(char_based_cap, line_based_cap)))

        preferred_floor = max(64, min(floor_max, min(ratio_based_floor, size_based_cap)))
        hard_floor = max(64, int(round(preferred_floor * 0.67)))
        hard_floor = min(hard_floor, preferred_floor)
        return {
            "agentocr_mm_preferred_visual_floor": int(preferred_floor),
            "agentocr_mm_hard_visual_floor": int(hard_floor),
            "agentocr_mm_ratio_visual_floor": int(ratio_based_floor),
            "agentocr_mm_size_cap_visual_floor": int(size_based_cap),
            "agentocr_trust_line_keep_ratio": float(line_keep_ratio),
            "agentocr_trust_char_keep_ratio": float(char_keep_ratio),
            "agentocr_trust_rendered_line_count": float(rendered_line_count),
            "agentocr_trust_raw_line_count": float(raw_line_count),
            "agentocr_trust_rendered_char_count": float(rendered_char_count),
            "agentocr_trust_raw_char_count": float(raw_char_count),
        }

    def _attach_metadata_to_array(self, image_array: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        if not isinstance(image_array, np.ndarray) or not metadata:
            return image_array
        wrapped = image_array if isinstance(image_array, OCRMetadataArray) else image_array.view(OCRMetadataArray)
        existing = dict(getattr(wrapped, "_agentocr_metadata", {}) or {})
        existing.update(metadata)
        wrapped._agentocr_metadata = existing
        return wrapped

    def _attach_trust_policy_mm_metadata(
        self,
        image_arrays: List[Union[np.ndarray, List[np.ndarray]]],
        *,
        qwen3_history_pages: bool,
    ) -> List[Union[np.ndarray, List[np.ndarray]]]:
        if not self._last_trust_policy_diagnostics:
            return image_arrays

        decorated: List[Union[np.ndarray, List[np.ndarray]]] = []
        for index, image_array in enumerate(image_arrays):
            metadata = self._compute_trust_policy_mm_floor_metadata(
                self._last_trust_policy_diagnostics[index]
                if index < len(self._last_trust_policy_diagnostics)
                else None
            )
            if not metadata:
                decorated.append(image_array)
                continue
            if qwen3_history_pages and isinstance(image_array, list):
                decorated.append([self._attach_metadata_to_array(page, metadata) for page in image_array])
            else:
                decorated.append(self._attach_metadata_to_array(image_array, metadata))
        return decorated
    
    def _find_matching_segments(self, context: str, env_idx: int) -> Optional[Tuple[List[str], List[Tuple[int, int]], List[Dict], int]]:
        """
        Find matching segments in cache for incremental rendering.
        Supports sliding window by matching individual segments rather than full prefixes.
        Segments are split by newlines (\n) to match memory structure.
        
        Args:
            context: Current trajectory context
            env_idx: Environment index
            
        Returns:
            (matched_segments, matched_ranges, matched_seg_infos, total_height) if found, None otherwise
            - matched_segments: List of matched segment texts (lines)
            - matched_ranges: List of (start_h, end_h) tuples for each matched segment
            - matched_seg_infos: List of segment info dicts (includes padding info)
            - total_height: Total height after all matched segments
        """
        if self._master_images is None or env_idx not in self._master_images:
            return None
        
        master_data = self._master_images[env_idx]
        segments = master_data.get('segments', [])
        
        if not segments:
            return None
        
        # Split context into segments by newlines (to match memory structure)
        context_segments = [line.strip() for line in context.split('\n') if line.strip()]
        
        if not context_segments:
            return None
        
        # Try to match segments from the beginning
        matched_segments = []
        matched_ranges = []
        matched_seg_infos = []
        
        for ctx_seg in context_segments:
            ctx_seg_hash = hash(ctx_seg)
            
            # Find matching segment in cache
            found = False
            for seg_info in segments:
                if seg_info['content_hash'] == ctx_seg_hash:
                    matched_segments.append(ctx_seg)
                    matched_ranges.append((seg_info['start_h'], seg_info['end_h']))
                    matched_seg_infos.append(seg_info)
                    found = True
                    break
            
            if not found:
                # No more consecutive matches, stop here
                break
        
        if matched_segments:
            # Calculate total height
            total_height = matched_ranges[-1][1] if matched_ranges else 0
            return (matched_segments, matched_ranges, matched_seg_infos, total_height)
        
        return None
    
    def _render_segment(
        self,
        segment_text: str,
        **override_kwargs
    ) -> np.ndarray:
        """
        Render a single segment to an image.
        
        This is the deterministic segment renderer R(l; psi) described in the paper.
        
        Args:
            segment_text: The segment text to render
            **override_kwargs: Rendering parameters (font, size, padding, width bound)
            
        Returns:
            Rendered segment as numpy array
        """
        # Render the segment without padding (padding added later at assembly)
        render_kwargs = {**override_kwargs, 'padding': 0, 'min_height': 0}
        img = self._convert_single(segment_text, self._get_config(**render_kwargs))
        return np.array(img) if img is not None else self._get_blank_array(**render_kwargs)
    
    def _convert_incremental(
        self,
        trajectory_contexts: List[str],
        current_steps: List[int],
        env_indices: List[int],
        batch_size: int,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Convert trajectory texts to images using segment-level caching.
        
        This implements the AgentOCR caching strategy described in the paper:
        1. Split history into segments: Split(h_t) = (l_1, ..., l_K)
        2. For each segment l_i:
           - Query cache: if k(l_i) in cache, get cached image
           - Otherwise, render with R(l_i; psi) and insert into cache
        3. Stack all segment images to get final image: I_t = Stack(I(l_i))
        
        Key benefits:
        - Each unique segment is rendered at most once
        - Segments can be reused whenever they match cached content
        - Supports sliding windows and repeated observations/actions
        
        Args:
            trajectory_contexts: List of trajectory text strings (h_t for each env)
            current_steps: List of current step numbers for each environment
            env_indices: List of real environment indices (for active_masks support)
            batch_size: Batch size for stats printing
            **override_kwargs: Override configuration parameters (psi)
        
        Returns:
            List of numpy arrays representing the assembled images (I_t for each env)
        """
        # Initialize segment caches if needed
        if self._segment_caches is None:
            self._segment_caches = {}
        
        image_arrays = []
        
        # Batch-level statistics accumulators
        batch_total_segments = 0
        batch_total_hits = 0
        batch_total_misses = 0
        
        for real_env_idx, context, current_step in zip(env_indices, trajectory_contexts, current_steps):
            self._cache_stats['total'] += 1
            
            context = context.strip() if context else ""
            if not context:
                # Empty context, return blank
                self._cache_stats['misses'] += 1
                image_arrays.append(self._get_blank_array(**override_kwargs))
                continue
            
            # Initialize segment cache for this environment if needed
            if real_env_idx not in self._segment_caches:
                self._segment_caches[real_env_idx] = SegmentCache()
            
            segment_cache = self._segment_caches[real_env_idx]
            
            # Step 1: Split history into segments
            # Split(h_t) = (l_1, ..., l_K)
            segments = split_into_segments(context)
            
            if not segments:
                self._cache_stats['misses'] += 1
                image_arrays.append(self._get_blank_array(**override_kwargs))
                continue
            
            # Step 2: For each segment, lookup cache or render
            segment_images: List[np.ndarray] = []
            local_hits = 0
            local_misses = 0
            
            for segment_text in segments:
                self._segment_cache_stats['total_segments'] += 1
                
                # Cache lookup: C[k(l_i)]
                cached_img = segment_cache.lookup(segment_text)
                
                if cached_img is not None:
                    # Cache hit! Reuse cached segment image
                    segment_images.append(cached_img)
                    local_hits += 1
                    self._segment_cache_stats['cache_hits'] += 1
                    self._segment_cache_stats['segments_reused'] += 1
                else:
                    # Cache miss - render segment with R(l_i; psi)
                    rendered_img = self._render_segment(segment_text, **override_kwargs)
                    
                    # Insert into cache: C[k(l_i)] <- I(l_i)
                    segment_cache.insert(segment_text, rendered_img)
                    
                    segment_images.append(rendered_img)
                    local_misses += 1
                    self._segment_cache_stats['cache_misses'] += 1
                    self._segment_cache_stats['segments_rendered'] += 1
            
            # Update overall cache stats based on segment-level results
            # If any segments were reused, count as partial hit
            if local_hits > 0:
                self._cache_stats['hits'] += 1
            else:
                self._cache_stats['misses'] += 1
            
            # Step 3: Assemble full image by stacking segment images
            # I_t = Stack(I(l_i))_{i=1}^{K}
            if len(segment_images) == 1:
                assembled_image = segment_images[0].copy()
            else:
                assembled_image = np.vstack(segment_images)
            
            image_arrays.append(assembled_image)
            
            # Accumulate batch-level stats
            batch_total_segments += len(segments)
            batch_total_hits += local_hits
            batch_total_misses += local_misses
        
        # Print batch-level cache statistics (once per batch)
        if batch_total_segments > 0:
            self._print_batch_segment_cache_stats(
                batch_size=len(trajectory_contexts),
                batch_segments=batch_total_segments,
                batch_hits=batch_total_hits,
                batch_misses=batch_total_misses
            )
        
        return image_arrays

    def _convert_incremental_segments(
        self,
        trajectory_segments: List[List[Union[str, Dict[str, str]]]],
        env_indices: List[int],
        batch_size: int,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Convert pre-segmented trajectory history to images using segment-level caching.

        Unlike `_convert_incremental`, the caller decides the segment boundaries.
        This is used by Qwen-specific structured OCR layouts so they can keep their
        block-oriented rendering while still reusing the existing segment cache.
        """
        if self._segment_caches is None:
            self._segment_caches = {}

        image_arrays = []
        batch_total_segments = 0
        batch_total_hits = 0
        batch_total_misses = 0

        for real_env_idx, segments in zip(env_indices, trajectory_segments):
            self._cache_stats['total'] += 1

            normalized_segments: List[Tuple[str, str]] = []
            for segment in segments or []:
                if isinstance(segment, dict):
                    render_text = str(segment.get("text", "") or "").strip()
                    cache_key_text = str(segment.get("cache_key", render_text) or "").strip()
                else:
                    render_text = str(segment or "").strip()
                    cache_key_text = render_text
                if render_text:
                    normalized_segments.append((render_text, cache_key_text))
            if not normalized_segments:
                self._cache_stats['misses'] += 1
                image_arrays.append(self._get_blank_array(**override_kwargs))
                continue

            if real_env_idx not in self._segment_caches:
                self._segment_caches[real_env_idx] = SegmentCache()

            segment_cache = self._segment_caches[real_env_idx]
            segment_images: List[np.ndarray] = []
            local_hits = 0
            local_misses = 0

            for segment_text, cache_key_text in normalized_segments:
                self._segment_cache_stats['total_segments'] += 1
                cached_img = segment_cache.lookup(segment_text, cache_key_text=cache_key_text)

                if cached_img is not None:
                    segment_images.append(cached_img)
                    local_hits += 1
                    self._segment_cache_stats['cache_hits'] += 1
                    self._segment_cache_stats['segments_reused'] += 1
                else:
                    rendered_img = self._render_segment(segment_text, **override_kwargs)
                    segment_cache.insert(segment_text, rendered_img, cache_key_text=cache_key_text)
                    segment_images.append(rendered_img)
                    local_misses += 1
                    self._segment_cache_stats['cache_misses'] += 1
                    self._segment_cache_stats['segments_rendered'] += 1

            if local_hits > 0:
                self._cache_stats['hits'] += 1
            else:
                self._cache_stats['misses'] += 1

            assembled_image = segment_images[0].copy() if len(segment_images) == 1 else np.vstack(segment_images)
            image_arrays.append(assembled_image)

            batch_total_segments += len(normalized_segments)
            batch_total_hits += local_hits
            batch_total_misses += local_misses

        if batch_total_segments > 0:
            self._print_batch_segment_cache_stats(
                batch_size=batch_size,
                batch_segments=batch_total_segments,
                batch_hits=batch_total_hits,
                batch_misses=batch_total_misses,
            )

        return image_arrays
    
    def _convert_incremental_compact(
        self,
        trajectory_contexts: List[str],
        current_steps: List[int],
        env_indices: List[int],
        batch_size: int,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Convert trajectory texts to images using compact mode with incremental caching.
        
        In compact mode:
        - Newlines are replaced with colored symbols (e.g., ⏎)
        - All content is treated as a single paragraph
        - Line wrapping happens due to fixed width
        - Complete lines (filled to width) are cached as images
        - Incomplete lines are kept as text and prepended to next render
        
        Caching Strategy:
        - Track the text that corresponds to cached complete lines
        - If new context starts with cached text, reuse cached image
        - Only render new content (incomplete text + new additions)
        
        Args:
            trajectory_contexts: List of trajectory text strings
            current_steps: List of current step numbers for each environment
            env_indices: List of real environment indices (for active_masks support)
            **override_kwargs: Override configuration parameters
        
        Returns:
            List of numpy arrays representing the images
        """
        if self._compact_cache is None:
            self._compact_cache = {}
        
        config = self._get_config(**override_kwargs)
        image_arrays = []
        
        for real_env_idx, context, current_step in zip(env_indices, trajectory_contexts, current_steps):
            self._cache_stats['total'] += 1
            self._compact_cache_stats['total'] += 1
            
            context = context.strip() if context else ""
            if not context:
                self._cache_stats['misses'] += 1
                self._compact_cache_stats['misses'] += 1
                image_arrays.append(self._get_blank_array(**override_kwargs))
                continue
            
            # Initialize compact cache for this environment if needed
            if real_env_idx not in self._compact_cache:
                self._compact_cache[real_env_idx] = {
                    'complete_lines_img': None,
                    'complete_lines_count': 0,
                    'last_full_compact_text': '',  # Full compact text from last render
                    'incomplete_text': '',          # Remaining text (didn't fill a line)
                    'last_context_hash': None
                }
            
            cache_data = self._compact_cache[real_env_idx]
            context_hash = hash(context)
            
            # Apply compact mode transformation to get the full compact text
            compact_text = apply_compact_mode(context, config['compact_symbol'])
            
            # Check if this is the exact same context (full cache hit)
            if cache_data['last_context_hash'] == context_hash and cache_data['complete_lines_img'] is not None:
                self._cache_stats['hits'] += 1
                self._compact_cache_stats['full_hits'] += 1
                self._compact_cache_stats['cached_lines_reused'] += cache_data['complete_lines_count']
                # Reconstruct from cached complete lines + incomplete portion
                result = self._render_compact_with_cache(real_env_idx, context, config)
                image_arrays.append(result)
                continue
            
            # Check if new context EXTENDS the cached content (incremental hit)
            # We check if the new compact_text starts with the last full compact_text
            last_compact_text = cache_data.get('last_full_compact_text', '')
            can_reuse_cache = (
                cache_data['complete_lines_img'] is not None and
                last_compact_text and
                compact_text.startswith(last_compact_text)
            )
            
            if can_reuse_cache:
                # Incremental update: reuse cached complete lines, only render new content
                self._cache_stats['hits'] += 1
                self._compact_cache_stats['partial_hits'] += 1
                self._compact_cache_stats['cached_lines_reused'] += cache_data['complete_lines_count']
                
                # Get font metrics
                font = _get_cached_font(config['font_path'], config['font_size'])
                _, line_height = get_font_metrics(font, config['font_size'])
                
                # The new content is everything after the last full compact text
                new_addition = compact_text[len(last_compact_text):].strip()
                
                # Text to render = incomplete_text from before + new_addition
                if cache_data['incomplete_text']:
                    text_to_render = cache_data['incomplete_text'] + ' ' + new_addition
                else:
                    text_to_render = new_addition
                text_to_render = text_to_render.strip()
                
                if text_to_render:
                    # Render only the new content
                    new_img, new_complete_lines, new_incomplete_text, new_lines = text_to_adaptive_image_compact(
                        text_to_render,
                        font_size=config['font_size'],
                        padding=0,
                        bg_color=config['bg_color'],
                        text_color=config['text_color'],
                        font_path=config['font_path'],
                        min_width=config['min_width'],
                        max_width=config['max_width'],
                        min_height=0,
                        max_height=config['max_height'],
                        use_precise=config['use_precise'],
                        compact_symbol=config['compact_symbol'],
                        highlight_configs=config['highlight_configs']
                    )
                    new_img_array = np.array(new_img)
                    
                    # Track newly rendered lines
                    self._compact_cache_stats['lines_rendered'] += len(new_lines)
                    
                    # Combine cached complete lines with newly rendered content
                    combined = np.vstack([cache_data['complete_lines_img'], new_img_array])
                    
                    # Update cache
                    if new_complete_lines > 0:
                        new_complete_height = new_complete_lines * line_height
                        
                        # New cached image = old cached + new complete lines portion
                        total_cached_height = cache_data['complete_lines_img'].shape[0] + new_complete_height
                        cache_data['complete_lines_img'] = combined[:total_cached_height, :, :].copy()
                        cache_data['complete_lines_count'] += new_complete_lines
                    
                    cache_data['incomplete_text'] = new_incomplete_text
                    cache_data['last_full_compact_text'] = compact_text  # Store the full compact text
                    cache_data['last_context_hash'] = context_hash
                    
                    image_arrays.append(combined)
                else:
                    # No new content, just use cached (shouldn't happen often)
                    cache_data['last_full_compact_text'] = compact_text
                    cache_data['last_context_hash'] = context_hash
                    result = self._render_compact_with_cache(real_env_idx, context, config)
                    image_arrays.append(result)
            else:
                # Cache miss or context doesn't extend cached content - full re-render
                self._cache_stats['misses'] += 1
                
                # Render the full compact text
                img, num_complete_lines, incomplete_text, lines = text_to_adaptive_image_compact(
                    compact_text,
                    font_size=config['font_size'],
                    padding=0,
                    bg_color=config['bg_color'],
                    text_color=config['text_color'],
                    font_path=config['font_path'],
                    min_width=config['min_width'],
                    max_width=config['max_width'],
                    min_height=0,
                    max_height=config['max_height'],
                    use_precise=config['use_precise'],
                    compact_symbol=config['compact_symbol'],
                    highlight_configs=config['highlight_configs']
                )
                
                img_array = np.array(img)
                
                # Get font metrics for height calculations
                font = _get_cached_font(config['font_path'], config['font_size'])
                _, line_height = get_font_metrics(font, config['font_size'])
                
                # Track lines rendered
                self._compact_cache_stats['lines_rendered'] += len(lines)
                
                # Update cache with complete lines
                if num_complete_lines > 0:
                    complete_height = num_complete_lines * line_height
                    cache_data['complete_lines_img'] = img_array[:complete_height, :, :].copy()
                    cache_data['complete_lines_count'] = num_complete_lines
                    # This is a real miss (had to re-render with cacheable content)
                    self._compact_cache_stats['misses'] += 1
                else:
                    cache_data['complete_lines_img'] = None
                    cache_data['complete_lines_count'] = 0
                    # Content too short to fill a complete line - not a cache failure
                    self._compact_cache_stats['no_complete_lines'] += 1
                
                cache_data['incomplete_text'] = incomplete_text
                cache_data['last_full_compact_text'] = compact_text  # Store full compact text for next comparison
                cache_data['last_context_hash'] = context_hash
                
                image_arrays.append(img_array)
        
        # Print compact cache stats when we've processed a new batch
        if batch_size > 0:
            current_batch = self._compact_cache_stats['total'] // batch_size
            if current_batch > self._last_printed_cache_batch:
                self._last_printed_cache_batch = current_batch
                self._print_compact_cache_stats()
        
        return image_arrays
    
    def _render_compact_with_cache(
        self,
        env_idx: int,
        context: str,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Render compact mode image using cached complete lines.
        
        Args:
            env_idx: Environment index
            context: Current context text
            config: Configuration dictionary
        
        Returns:
            Rendered image as numpy array
        """
        cache_data = self._compact_cache[env_idx]
        
        # If no cached complete lines, render from scratch
        if cache_data['complete_lines_img'] is None:
            img, _, incomplete_text, _ = text_to_adaptive_image_compact(
                context,
                font_size=config['font_size'],
                padding=0,
                bg_color=config['bg_color'],
                text_color=config['text_color'],
                font_path=config['font_path'],
                min_width=config['min_width'],
                max_width=config['max_width'],
                min_height=0,
                max_height=config['max_height'],
                use_precise=config['use_precise'],
                compact_symbol=config['compact_symbol'],
                highlight_configs=config['highlight_configs']
            )
            cache_data['incomplete_text'] = incomplete_text
            return np.array(img)
        
        # Render only the incomplete portion and combine with cached complete lines
        incomplete_text = cache_data['incomplete_text']
        
        if incomplete_text:
            # Render the incomplete text
            img, _, new_incomplete, _ = text_to_adaptive_image_compact(
                incomplete_text,
                font_size=config['font_size'],
                padding=0,
                bg_color=config['bg_color'],
                text_color=config['text_color'],
                font_path=config['font_path'],
                min_width=config['min_width'],
                max_width=config['max_width'],
                min_height=0,
                max_height=config['max_height'],
                use_precise=config['use_precise'],
                compact_symbol=config['compact_symbol'],
                highlight_configs=config['highlight_configs']
            )
            incomplete_img = np.array(img)
            
            # Combine cached complete lines with incomplete line render
            combined = np.vstack([cache_data['complete_lines_img'], incomplete_img])
            return combined
        else:
            # No incomplete text, just return cached complete lines
            return cache_data['complete_lines_img'].copy()
    
    def _update_master_image(self, env_idx: int, context: str, context_hash: int,
                            new_img: np.ndarray, line_ranges: Optional[List[Tuple[int, int]]],
                            step_start: int, step_end: int,
                            **override_kwargs):
        """
        Update master image for an environment by appending new content.
        Stores individual segments (lines split by \n) to support sliding window matching.
        
        Optimized strategy: Directly use the pre-rendered image without re-rendering.
        Each new_content (already rendered) is treated as one or more segments.
        
        Args:
            env_idx: Environment index
            context: Full context string (used to extract line segments)
            context_hash: Hash of context (for backward compatibility)
            new_img: Pre-rendered image to append (already rendered, no re-rendering needed)
            step_start: Starting step number of this context
            step_end: Ending step number of this context
        """
        master_data = self._master_images[env_idx]
        
        # Initialize segments list if needed
        if 'segments' not in master_data:
            master_data['segments'] = []
        if 'indices' not in master_data:
            master_data['indices'] = {}  # Keep for backward compatibility
        
        # Split context into segments by newlines (to match memory structure)
        context_lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        # Append the pre-rendered image to master image
        if master_data['master_img'] is None:
            master_data['master_img'] = new_img
            start_h = 0
            end_h = new_img.shape[0]
        else:
            start_h = master_data['master_img'].shape[0]
            master_data['master_img'] = np.vstack([master_data['master_img'], new_img])
            end_h = master_data['master_img'].shape[0]
        
        # Store each line as a separate segment for cache matching.
        # If line_ranges is provided, use precise heights per line; otherwise fall back to the whole block.
        if line_ranges:
            ranges_iter = [(start_h + s, start_h + e) for (s, e) in line_ranges]
        else:
            ranges_iter = [(start_h, end_h)] * max(len(context_lines), 1)
        for line, (seg_start, seg_end) in zip(context_lines, ranges_iter):
            line_hash = hash(line)
            exists = any(seg['content_hash'] == line_hash for seg in master_data['segments'])
            if not exists:
                master_data['segments'].append({
                    'content_hash': line_hash,
                    'step': step_end,
                    'start_h': seg_start,
                    'end_h': seg_end,
                    'text': line
                })
        
        # Store index for backward compatibility (for exact context matching)
        master_data['indices'][context_hash] = (start_h, end_h, step_start, step_end)
        
    
    
    def _print_cache_stats(self):
        """Print cache hit rate statistics."""
        stats = self._cache_stats
        total = stats['total']
        hits = stats['hits']
        misses = stats['misses']
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        print(f"[OCR Cache] Total: {total}, Hits: {hits}, Misses: {misses}, Hit Rate: {hit_rate:.1f}%")
    
    def _print_batch_segment_cache_stats(
        self,
        batch_size: int,
        batch_segments: int,
        batch_hits: int,
        batch_misses: int
    ):
        """
        Print batch-level segment cache statistics (once per batch).
        
        Args:
            batch_size: Number of environments in this batch
            batch_segments: Total segments processed in this batch
            batch_hits: Total cache hits in this batch
            batch_misses: Total cache misses in this batch
        """
        # Batch statistics
        batch_reuse_rate = (batch_hits / batch_segments * 100) if batch_segments > 0 else 0
        
        # Cumulative statistics
        cum_stats = self._segment_cache_stats
        cum_total = cum_stats['total_segments']
        cum_hits = cum_stats['cache_hits']
        cum_rendered = cum_stats['segments_rendered']
        cum_reused = cum_stats['segments_reused']
        cum_hit_rate = (cum_hits / cum_total * 100) if cum_total > 0 else 0
        cum_savings = (cum_reused / (cum_rendered + cum_reused) * 100) if (cum_rendered + cum_reused) > 0 else 0
        
        # Total cache status across all environments
        total_cached = 0
        total_cache_size_mb = 0.0
        num_envs = 0
        if self._segment_caches:
            num_envs = len(self._segment_caches)
            for cache in self._segment_caches.values():
                total_cached += len(cache)
                total_cache_size_mb += cache.get_stats()['cache_size_mb']
        
        print(f"[OCR Render] Batch Size={batch_size} | "
              f"Num_Segments={batch_segments} | "
              f"Num_Hit={batch_hits} | "
              f"Num_Miss={batch_misses} | "
              f"Batch Hit Rate={batch_reuse_rate:.1f}% || "
              f"Cached_Segments={total_cached} | "
              f"Cache_Size={total_cache_size_mb:.2f}MB")
    
    def _print_segment_cache_stats(self):
        """Print segment-level cache statistics (summary)."""
        stats = self._segment_cache_stats
        total_segments = stats['total_segments']
        cache_hits = stats['cache_hits']
        cache_misses = stats['cache_misses']
        segments_rendered = stats['segments_rendered']
        segments_reused = stats['segments_reused']
        
        # Calculate hit rate at segment level
        segment_hit_rate = (cache_hits / total_segments * 100) if total_segments > 0 else 0
        
        # Calculate rendering savings
        total_would_render = segments_rendered + segments_reused
        savings_rate = (segments_reused / total_would_render * 100) if total_would_render > 0 else 0
        
        # Count total cached segments across all environments
        total_cached = 0
        total_cache_size_mb = 0.0
        if self._segment_caches:
            for cache in self._segment_caches.values():
                total_cached += len(cache)
                cache_stats = cache.get_stats()
                total_cache_size_mb += cache_stats['cache_size_mb']
        
        print(f"[OCR Segment Cache Summary] Segments: {total_segments} | "
              f"Hits: {cache_hits} ({segment_hit_rate:.1f}%) | "
              f"Misses: {cache_misses} | "
              f"Rendered: {segments_rendered} | "
              f"Reused: {segments_reused} | "
              f"Savings: {savings_rate:.1f}%")
        print(f"[OCR Segment Cache Summary] Cached Segments: {total_cached} | "
              f"Cache Size: {total_cache_size_mb:.2f} MB")
    
    def get_cache_stats(self):
        """Get cache statistics."""
        stats = self._cache_stats
        total = stats['total']
        hits = stats['hits']
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'hits': hits,
            'misses': stats['misses'],
            'hit_rate': f'{hit_rate:.1f}%'
        }
    
    def get_segment_cache_stats(self) -> Dict[str, Any]:
        """
        Get segment-level cache statistics.
        
        Returns:
            Dictionary with segment cache statistics including:
            - total_segments: Total segment lookups
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - segment_hit_rate: Cache hit rate percentage
            - segments_rendered: Number of segments actually rendered
            - segments_reused: Number of segments reused from cache
            - savings_rate: Rendering savings percentage
            - total_cached: Total unique segments in cache
            - cache_size_mb: Total cache memory usage in MB
        """
        stats = self._segment_cache_stats
        total_segments = stats['total_segments']
        cache_hits = stats['cache_hits']
        segments_rendered = stats['segments_rendered']
        segments_reused = stats['segments_reused']
        
        segment_hit_rate = (cache_hits / total_segments * 100) if total_segments > 0 else 0
        total_would_render = segments_rendered + segments_reused
        savings_rate = (segments_reused / total_would_render * 100) if total_would_render > 0 else 0
        
        # Aggregate across all environment caches
        total_cached = 0
        total_cache_size_mb = 0.0
        if self._segment_caches:
            for cache in self._segment_caches.values():
                total_cached += len(cache)
                cache_stats = cache.get_stats()
                total_cache_size_mb += cache_stats['cache_size_mb']
        
        return {
            'total_segments': total_segments,
            'cache_hits': cache_hits,
            'cache_misses': stats['cache_misses'],
            'segment_hit_rate': f'{segment_hit_rate:.1f}%',
            'segments_rendered': segments_rendered,
            'segments_reused': segments_reused,
            'savings_rate': f'{savings_rate:.1f}%',
            'total_cached': total_cached,
            'cache_size_mb': f'{total_cache_size_mb:.2f}',
        }
    
    def _print_compact_cache_stats(self):
        """Print compact mode cache statistics with detailed breakdown."""
        stats = self._compact_cache_stats
        total = stats['total']
        full_hits = stats['full_hits']
        partial_hits = stats['partial_hits']
        misses = stats['misses']
        no_complete = stats['no_complete_lines']
        cached_lines = stats['cached_lines_reused']
        rendered_lines = stats['lines_rendered']
        
        # Calculate rates (exclude no_complete_lines from miss rate since it's not a cache failure)
        cacheable_total = total - no_complete
        full_hit_rate = (full_hits / cacheable_total * 100) if cacheable_total > 0 else 0
        partial_hit_rate = (partial_hits / cacheable_total * 100) if cacheable_total > 0 else 0
        total_hit_rate = ((full_hits + partial_hits) / cacheable_total * 100) if cacheable_total > 0 else 0
        
        # Calculate line-level savings
        total_lines = cached_lines + rendered_lines
        line_savings = (cached_lines / total_lines * 100) if total_lines > 0 else 0
        
        print(f"[OCR Compact Cache] Total: {total} | "
              f"Full Hits: {full_hits} ({full_hit_rate:.1f}%) | "
              f"Partial Hits: {partial_hits} ({partial_hit_rate:.1f}%) | "
              f"Misses: {misses} | "
              f"NoCache: {no_complete} | "
              f"Hit Rate: {total_hit_rate:.1f}%")
        print(f"[OCR Compact Cache] Lines Reused: {cached_lines} | "
              f"Lines Rendered: {rendered_lines} | "
              f"Line Savings: {line_savings:.1f}%")
    
    def get_compact_cache_stats(self):
        """Get compact mode cache statistics."""
        stats = self._compact_cache_stats
        total = stats['total']
        full_hits = stats['full_hits']
        partial_hits = stats['partial_hits']
        no_complete = stats['no_complete_lines']
        cached_lines = stats['cached_lines_reused']
        rendered_lines = stats['lines_rendered']
        
        # Calculate hit rate excluding non-cacheable requests
        cacheable_total = total - no_complete
        total_hit_rate = ((full_hits + partial_hits) / cacheable_total * 100) if cacheable_total > 0 else 0
        total_lines = cached_lines + rendered_lines
        line_savings = (cached_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            'total': total,
            'full_hits': full_hits,
            'partial_hits': partial_hits,
            'misses': stats['misses'],
            'no_complete_lines': no_complete,
            'hit_rate': f'{total_hit_rate:.1f}%',
            'cached_lines_reused': cached_lines,
            'lines_rendered': rendered_lines,
            'line_savings': f'{line_savings:.1f}%'
        }

    def _build_qwen3_history_block(
        self,
        step_num: int,
        action_text: str = "",
        observation_text: str = "",
        raw_lines: Optional[List[str]] = None,
        compact_layout: bool = False,
    ) -> Dict[str, Any]:
        step_prefix = f"[MEMORY {step_num:02d}]" if compact_layout else f"[STEP {step_num:02d}]"
        if compact_layout:
            block_lines: List[str] = []
            if action_text and observation_text:
                block_lines.append(f"{step_prefix} Action: {action_text}")
                block_lines.append(f"Observation: {observation_text}")
            elif action_text:
                block_lines.append(f"{step_prefix} Action: {action_text}")
            elif observation_text:
                block_lines.append(f"{step_prefix} Observation: {observation_text}")
        else:
            block_lines = [step_prefix]
            if action_text:
                block_lines.append(f"Action: {action_text}")
            if observation_text:
                block_lines.append(f"Observation: {observation_text}")
        if not action_text and not observation_text:
            if compact_layout and not block_lines:
                block_lines.append(step_prefix)
            for raw_line in raw_lines or []:
                cleaned_line = str(raw_line or "").strip()
                if cleaned_line:
                    block_lines.append(cleaned_line)

        raw_text_parts = []
        if action_text:
            raw_text_parts.append(f"[Action]: {action_text}")
        if observation_text:
            raw_text_parts.append(f"[Observation]: {observation_text}")
        if not raw_text_parts:
            raw_text_parts.extend(str(raw_line or "").strip() for raw_line in (raw_lines or []) if str(raw_line or "").strip())

        return {
            "step": step_num,
            "action_text": action_text,
            "observation_text": observation_text,
            "raw_line": " ".join(raw_text_parts).strip(),
            "lines": block_lines,
        }

    def _build_qwen3_search_history_block(
        self,
        step_num: int,
        raw_lines: List[str],
        compact_layout: bool = False,
    ) -> Dict[str, Any]:
        step_prefix = f"[MEMORY {step_num:02d}]" if compact_layout else f"[STEP {step_num:02d}]"
        action_text = ""
        observation_lines: List[str] = []

        for raw_line in raw_lines:
            cleaned_line = str(raw_line or "").strip()
            if not cleaned_line or re.match(r"^\[Step\s+\d+\]$", cleaned_line):
                continue
            if cleaned_line in {"<information>", "</information>"}:
                continue
            if not action_text and cleaned_line.startswith("<search>") and cleaned_line.endswith("</search>"):
                action_text = cleaned_line
                continue
            observation_lines.append(cleaned_line)

        block_lines: List[str] = [step_prefix]
        if action_text:
            block_lines.append(action_text)
        if observation_lines:
            # Keep retrieved evidence explicitly wrapped so the OCR image matches
            # the search prompt contract: queries live in <search> and evidence
            # lives in <information>. Without these wrappers the model can
            # overread query lines and underweight the actual retrieved docs.
            block_lines.append("<information>")
            block_lines.extend(observation_lines)
            block_lines.append("</information>")

        raw_text_parts: List[str] = []
        if action_text:
            raw_text_parts.append(action_text)
        raw_text_parts.extend(observation_lines)

        return {
            "step": step_num,
            "action_text": action_text,
            "observation_text": "\n".join(observation_lines).strip(),
            "raw_line": " ".join(raw_text_parts).strip(),
            "lines": block_lines,
        }

    def _parse_qwen3_search_history_blocks(
        self,
        history_lines: List[str],
        compact_layout: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        grouped_steps: List[Tuple[int, List[str]]] = []
        current_step_num: Optional[int] = None
        current_step_lines: List[str] = []
        saw_step_header = False

        for raw_line in history_lines:
            step_match = re.match(r"^\[Step\s+(\d+)\]$", raw_line)
            if step_match:
                saw_step_header = True
                if current_step_num is not None:
                    grouped_steps.append((current_step_num, current_step_lines))
                current_step_num = int(step_match.group(1))
                current_step_lines = [raw_line]
                continue

            if current_step_num is None:
                return None
            current_step_lines.append(raw_line)

        if not saw_step_header:
            return None

        if current_step_num is not None:
            grouped_steps.append((current_step_num, current_step_lines))

        if not grouped_steps:
            return None

        return [
            self._build_qwen3_search_history_block(
                step_num=step_num,
                raw_lines=step_lines,
                compact_layout=compact_layout,
            )
            for step_num, step_lines in grouped_steps
        ]

    def _split_qwen3_history_blocks(
        self,
        context: str,
        current_step: Optional[int] = None,
        compact_layout: bool = False,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        lines = [line.strip() for line in str(context or "").split("\n") if line.strip()]
        if not lines:
            return [], []

        skill_lines: List[str] = []
        history_lines: List[str] = []
        for raw_line in lines:
            if raw_line.startswith("[SKILL]"):
                skill_lines.append(raw_line)
                continue
            history_lines.append(raw_line)

        if not history_lines:
            return skill_lines, []

        search_blocks = self._parse_qwen3_search_history_blocks(
            history_lines,
            compact_layout=compact_layout,
        )
        if search_blocks is not None:
            return skill_lines, search_blocks

        if compact_layout:
            compact_blocks = [
                self._build_qwen3_history_block(
                    step_num=index,
                    raw_lines=[raw_line],
                    compact_layout=True,
                )
                for index, raw_line in enumerate(history_lines, start=1)
            ]
            return skill_lines, compact_blocks

        parsed_entries: List[Dict[str, Any]] = []
        pending_action = ""

        for raw_line in history_lines:
            combined_match = re.match(r"^\[Observation\]:\s*(.*?)\s+\[Action\]:\s*(.*)$", raw_line)
            if combined_match:
                if pending_action:
                    parsed_entries.append(
                        {
                            "action_text": pending_action,
                            "observation_text": "",
                            "raw_lines": [f"[Action]: {pending_action}"],
                        }
                    )
                    pending_action = ""
                parsed_entries.append(
                    {
                        "action_text": combined_match.group(2).strip(),
                        "observation_text": combined_match.group(1).strip(),
                        "raw_lines": [raw_line],
                    }
                )
                continue

            if raw_line.startswith("[Action]:"):
                action_text = raw_line[len("[Action]:"):].strip()
                if pending_action:
                    parsed_entries.append(
                        {
                            "action_text": pending_action,
                            "observation_text": "",
                            "raw_lines": [f"[Action]: {pending_action}"],
                        }
                    )
                pending_action = action_text
                continue

            if raw_line.startswith("[Observation]:"):
                observation_text = raw_line[len("[Observation]:"):].strip()
                parsed_entries.append(
                    {
                        "action_text": pending_action,
                        "observation_text": observation_text,
                        "raw_lines": [raw_line] if not pending_action else [f"[Action]: {pending_action}", raw_line],
                    }
                )
                pending_action = ""
                continue

            if pending_action:
                parsed_entries.append(
                    {
                        "action_text": pending_action,
                        "observation_text": "",
                        "raw_lines": [f"[Action]: {pending_action}"],
                    }
                )
                pending_action = ""
            parsed_entries.append(
                {
                    "action_text": "",
                    "observation_text": "",
                    "raw_lines": [raw_line],
                }
            )

        if pending_action:
            parsed_entries.append(
                {
                    "action_text": pending_action,
                    "observation_text": "",
                    "raw_lines": [f"[Action]: {pending_action}"],
                }
            )

        block_count = len(parsed_entries)
        terminal_step = int(current_step) if current_step is not None else block_count
        start_step = max(1, terminal_step - block_count + 1)
        blocks = [
            self._build_qwen3_history_block(
                step_num=start_step + offset,
                action_text=entry["action_text"],
                observation_text=entry["observation_text"],
                raw_lines=entry["raw_lines"],
                compact_layout=compact_layout,
            )
            for offset, entry in enumerate(parsed_entries)
        ]
        return skill_lines, blocks

    def _summarize_qwen3_history_snapshot(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        latest_block = self._qwen3_latest_history_block(blocks)
        latest_step = latest_block.get("step")
        latest_action = str(latest_block.get("action_text") or "")
        latest_observation = str(latest_block.get("observation_text") or "")
        return {
            "latest_step": latest_step,
            "latest_action": latest_action,
            "latest_observation": latest_observation,
        }

    def _parse_qwen3_skill_fields(self, body: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for match in re.finditer(r"([A-Za-z0-9_]+)=([^=]+?)(?=\s+[A-Za-z0-9_]+=|$)", str(body or "").strip()):
            key = str(match.group(1) or "").strip().lower()
            value = str(match.group(2) or "").strip()
            if key and value:
                fields[key] = value
        return fields

    def _collect_qwen3_snapshot_state(self, context: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "goal_targets": [],
            "goal_receptacle": "",
            "required_state": "",
            "state_device": "",
            "current_location": "",
            "visible_anchor": "",
            "target_locations": {},
            "object_progress": {},
            "searched_without_target": [],
            "state_flags": {},
        }
        for raw_line in context.splitlines():
            line = str(raw_line or "").strip()
            match = re.match(r"^\[SKILL\]\[(?P<kind>[^\]]+)\]\s*(?P<body>.*)$", line)
            if not match:
                continue
            kind = str(match.group("kind") or "").strip().lower()
            fields = self._parse_qwen3_skill_fields(match.group("body") or "")
            if kind == "goal":
                targets_value = fields.get("targets", "")
                state["goal_targets"] = [
                    token.strip().replace("_", " ")
                    for token in targets_value.split(",")
                    if token.strip() and token.strip() != "none"
                ]
                receptacle_value = str(fields.get("receptacle", "")).strip().replace("_", " ")
                state["goal_receptacle"] = "" if receptacle_value == "none" else receptacle_value
                required_state_value = str(fields.get("required_state", "")).strip()
                state["required_state"] = "" if required_state_value == "none" else required_state_value
                state_device_value = str(fields.get("state_device", "")).strip().replace("_", " ")
                state["state_device"] = "" if state_device_value == "none" else state_device_value
            elif kind == "location":
                state["current_location"] = (
                    ""
                    if fields.get("agent") == "unknown"
                    else str(fields.get("agent", "")).strip().replace("_", " ")
                )
                state["visible_anchor"] = (
                    ""
                    if fields.get("anchor") == "unknown"
                    else str(fields.get("anchor", "")).strip().replace("_", " ")
                )
                target_locations: Dict[str, str] = {}
                for key, value in fields.items():
                    if key in {"agent", "anchor", "conf"}:
                        continue
                    target_locations[key.replace("_", " ")] = value.replace("_", " ")
                state["target_locations"] = target_locations
            elif kind == "progress":
                object_progress: Dict[str, str] = {}
                searched_without_target: List[str] = []
                for key, value in fields.items():
                    if key == "searched_without_target":
                        searched_without_target = [
                            token.strip().replace("_", " ")
                            for token in value.split(",")
                            if token.strip()
                        ]
                    else:
                        object_progress[key.replace("_", " ")] = value
                state["object_progress"] = object_progress
                state["searched_without_target"] = searched_without_target
            elif kind == "state":
                state["state_flags"] = {
                    key.replace("_", " "): value
                    for key, value in fields.items()
                }
        return state

    def _infer_qwen3_snapshot_phase(self, snapshot_state: Dict[str, Any]) -> str:
        goal_targets = tuple(snapshot_state.get("goal_targets", ()))
        object_progress = dict(snapshot_state.get("object_progress", {}))
        progress_states = [object_progress.get(goal_object, "missing") for goal_object in goal_targets]
        if any(state == "placed" for state in progress_states):
            return "placement complete"
        if any(state == "holding" for state in progress_states):
            return "carrying target"
        if any(state == "located" for state in progress_states) or any(
            dict(snapshot_state.get("target_locations", {})).get(goal_object) for goal_object in goal_targets
        ):
            return "target located"
        if snapshot_state.get("searched_without_target"):
            return "searching"
        return "gathering evidence"

    def _build_qwen3_stable_snapshot_lines(
        self,
        context: str,
        *,
        include_skill_lines: bool,
        compact_layout: bool = False,
    ) -> List[str]:
        snapshot_state = self._collect_qwen3_snapshot_state(context)
        goal_targets = list(snapshot_state.get("goal_targets", []))
        goal_receptacle = str(snapshot_state.get("goal_receptacle", "")).strip()
        required_state = str(snapshot_state.get("required_state", "")).strip()
        state_device = str(snapshot_state.get("state_device", "")).strip()
        if not goal_targets and not goal_receptacle and not snapshot_state.get("searched_without_target"):
            return []
        phase_label = self._infer_qwen3_snapshot_phase(snapshot_state)

        object_progress = dict(snapshot_state.get("object_progress", {}))
        target_locations = dict(snapshot_state.get("target_locations", {}))
        searched_without_target = list(snapshot_state.get("searched_without_target", []))
        state_flags = dict(snapshot_state.get("state_flags", {}))
        normalized_required_state = required_state.strip().lower()
        desired_state_complete = False
        if normalized_required_state:
            relevant_state_subjects = {
                str(goal_object).strip().lower()
                for goal_object in goal_targets
                if str(goal_object).strip()
            }
            if state_device:
                relevant_state_subjects.add(state_device.strip().lower())
            desired_state_complete = any(
                str(state_name or "").strip().lower() == normalized_required_state
                and str(subject or "").strip().lower() in relevant_state_subjects
                for subject, state_name in state_flags.items()
            )

        if compact_layout:
            snapshot_lines: List[str] = ["[LATEST SNAPSHOT]"]
            if required_state:
                snapshot_lines.append(f"Phase: {phase_label}")
            if goal_targets:
                snapshot_lines.append(f"Target: {', '.join(goal_targets[:2])}")
            if goal_receptacle:
                snapshot_lines.append(f"Placement target: {goal_receptacle}")
            if required_state:
                snapshot_lines.append(f"Required state: {required_state}")
        else:
            snapshot_lines = ["[LATEST SNAPSHOT]", f"Phase: {phase_label}"]
            if goal_targets:
                snapshot_lines.append(f"Target: {', '.join(goal_targets[:2])}")
            if required_state:
                snapshot_lines.append(f"Required state: {required_state}")
            if goal_receptacle and (desired_state_complete or not required_state):
                snapshot_lines.append(f"Placement target: {goal_receptacle}")

        holding_targets = [
            goal_object
            for goal_object in goal_targets
            if object_progress.get(goal_object) == "holding"
        ]
        progress_lines = [
            f"Progress: {goal_object}={object_progress.get(goal_object)}"
            for goal_object in goal_targets
            if str(object_progress.get(goal_object) or "").strip()
        ]
        known_locations = [
            (goal_object, str(target_locations.get(goal_object) or "").strip())
            for goal_object in goal_targets
            if str(target_locations.get(goal_object) or "").strip()
            and object_progress.get(goal_object) != "holding"
        ]
        if state_device:
            state_device_location = str(target_locations.get(state_device) or "").strip()
            if state_device_location and all(subject != state_device for subject, _ in known_locations):
                known_locations.append((state_device, state_device_location))
        relevant_states = [
            f"{subject}={state_name}"
            for subject, state_name in sorted(state_flags.items())
            if state_name
            and (
                any(subject == goal_object for goal_object in goal_targets)
                or (goal_receptacle and subject == goal_receptacle)
                or (state_device and subject == state_device)
            )
        ]

        render_snapshot_details = (not include_skill_lines) or bool(required_state)
        if render_snapshot_details:
            if state_device and required_state and not desired_state_complete:
                snapshot_lines.append(f"State device: {state_device}")
            if holding_targets:
                snapshot_lines.append(f"Inventory: {', '.join(holding_targets[:2])}")
            for subject, location_value in known_locations[:2]:
                snapshot_lines.append(f"Known location: {subject} -> {location_value}")
            for progress_line in progress_lines[:2]:
                snapshot_lines.append(progress_line)
            if relevant_states:
                snapshot_lines.append(f"State: {', '.join(relevant_states[:3])}")
            if compact_layout and searched_without_target:
                snapshot_lines.append(
                    "Searched without target: " + ", ".join(searched_without_target[:3])
                )
            elif (not compact_layout) and phase_label == "searching" and searched_without_target:
                snapshot_lines.append(
                    "Searched without target: " + ", ".join(searched_without_target[:3])
                )

        if not compact_layout:
            next_focus = ""
            if required_state and not desired_state_complete:
                action_verb = {
                    "heated": "heat",
                    "cooled": "cool",
                    "cleaned": "clean",
                    "light_on": "turn on",
                }.get(required_state, required_state)
                if holding_targets and state_device:
                    next_focus = f"{action_verb} {holding_targets[0]} with {state_device}"
                elif holding_targets:
                    next_focus = f"{action_verb} {holding_targets[0]}"
                elif known_locations:
                    subject, location_value = known_locations[0]
                    next_focus = f"retrieve {subject} from {location_value}"
                elif goal_targets:
                    next_focus = f"continue search for {goal_targets[0]}"
            elif holding_targets and goal_receptacle:
                next_focus = f"place {holding_targets[0]} into {goal_receptacle}"
            elif holding_targets:
                next_focus = f"use carried target: {holding_targets[0]}"
            elif known_locations:
                subject, location_value = known_locations[0]
                next_focus = f"retrieve {subject} from {location_value}"
            elif goal_targets:
                next_focus = f"continue search for {goal_targets[0]}"
            if next_focus:
                snapshot_lines.append(f"Next focus: {next_focus}")
            if goal_receptacle and required_state and not desired_state_complete:
                snapshot_lines.append(f"Placement target: {goal_receptacle}")

        if required_state:
            max_snapshot_lines = 9
        elif not include_skill_lines:
            max_snapshot_lines = 8
        else:
            max_snapshot_lines = 4
        return snapshot_lines[:max_snapshot_lines]

    def _compute_qwen3_snapshot_overlap(
        self,
        snapshot_lines: List[str],
        latest_block_lines: List[str],
        goal_slots: GoalSlots,
    ) -> Dict[str, float]:
        snapshot_fact_keys: Set[str] = set()
        for line in snapshot_lines:
            stripped = str(line or "").strip()
            if stripped.startswith("Known location:") and "->" in stripped:
                payload = stripped.split(":", 1)[1].strip()
                subject, value = payload.split("->", 1)
                snapshot_fact_keys.add(
                    f"location|{' '.join(subject.strip().split())}|{' '.join(value.strip().split())}"
                )
            elif stripped.startswith("Inventory:"):
                for token in stripped.split(":", 1)[1].split(","):
                    subject = " ".join(token.strip().split())
                    if subject:
                        snapshot_fact_keys.add(f"progress|holding|{subject}")
            elif stripped.startswith("Progress:"):
                payload = stripped.split(":", 1)[1].strip()
                if "=" in payload:
                    subject, progress_value = payload.split("=", 1)
                    subject = " ".join(subject.strip().split())
                    progress_value = " ".join(progress_value.strip().split())
                    if subject and progress_value:
                        snapshot_fact_keys.add(f"progress|{subject}|{progress_value}")
            elif stripped.startswith("State:"):
                for token in stripped.split(":", 1)[1].split(","):
                    if "=" not in token:
                        continue
                    subject, state_name = token.split("=", 1)
                    snapshot_fact_keys.add(
                        f"state|{' '.join(subject.strip().split())}|{' '.join(state_name.strip().split())}"
                    )
            elif stripped.startswith("Required state:"):
                required_state = " ".join(stripped.split(":", 1)[1].strip().split())
                if required_state:
                    snapshot_fact_keys.add(f"required_state|goal|{required_state}")
            elif stripped.startswith("Searched without target:"):
                for token in stripped.split(":", 1)[1].split(","):
                    location = " ".join(token.strip().split())
                    if location:
                        snapshot_fact_keys.add(f"progress|searched_without_target|{location}")

        latest_fact_keys: Set[str] = set()
        for line in latest_block_lines:
            latest_fact_keys.update(_line_summary_fact_keys(line, kind=None, goal_slots=goal_slots))

        overlap = snapshot_fact_keys.intersection(latest_fact_keys)
        overlap_count = float(len(overlap))
        overlap_rate = overlap_count / float(len(snapshot_fact_keys)) if snapshot_fact_keys else 0.0
        return {
            "snapshot_rendered_line_count": float(len(snapshot_fact_keys)),
            "snapshot_latest_block_overlap_count": overlap_count,
            "snapshot_latest_block_overlap_rate": overlap_rate,
        }

    def _qwen3_page_prefix_lines(
        self,
        header: str,
        page_index: int,
        snapshot_lines: List[str],
        *,
        compact_layout: bool = False,
        history_order_label: str = "NEWEST FIRST",
    ) -> List[str]:
        prefix_lines = [header]
        if page_index != 0:
            return prefix_lines
        if snapshot_lines:
            prefix_lines.extend(["", *snapshot_lines])
        if not compact_layout:
            prefix_lines.extend(["", f"[RECENT HISTORY | {history_order_label}]"])
        return prefix_lines

    def _qwen3_history_blocks_newest_first(self, blocks: List[Dict[str, Any]]) -> bool:
        if len(blocks) < 2:
            return True
        try:
            return int(blocks[0].get("step", 0) or 0) >= int(blocks[-1].get("step", 0) or 0)
        except Exception:
            return True

    def _qwen3_latest_history_block(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not blocks:
            return {"lines": [], "step": None, "action_text": "", "observation_text": ""}
        return max(blocks, key=lambda block: int(block.get("step", -1) or -1))

    def _normalize_qwen3_cache_key_text(self, text: str) -> str:
        normalized_lines = [str(line or "").rstrip() for line in str(text or "").splitlines()]
        return "\n".join(normalized_lines).strip()

    def _build_qwen3_cache_segment(
        self,
        text: str,
        *,
        block_name: str,
    ) -> Optional[Dict[str, str]]:
        render_text = str(text or "").strip()
        if not render_text:
            return None
        normalized = self._normalize_qwen3_cache_key_text(render_text)
        return {
            "text": render_text,
            "cache_key": f"{block_name}\n{normalized}",
            "block_name": block_name,
        }

    def _qwen3_snapshot_cache_block_name(self, line: str) -> str:
        stripped = str(line or "").strip()
        if not stripped:
            return "snapshot/blank"
        if stripped.startswith("[LATEST SNAPSHOT]"):
            return "snapshot/header"
        if ":" not in stripped:
            return "snapshot/misc"
        label = stripped.split(":", 1)[0].strip().lower()
        label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
        return f"snapshot/{label or 'misc'}"

    def _qwen3_skill_cache_block_name(self, line: str) -> str:
        match = re.match(r"^\[SKILL\]\[(?P<kind>[^\]]+)\]", str(line or "").strip())
        if not match:
            return "skill/misc"
        kind = re.sub(r"[^a-z0-9]+", "_", str(match.group("kind") or "").strip().lower()).strip("_")
        return f"skill/{kind or 'misc'}"

    def _qwen3_skill_kind(self, line: str) -> str:
        match = re.match(r"^\[SKILL\]\[(?P<kind>[^\]]+)\]", str(line or "").strip())
        if not match:
            return ""
        return str(match.group("kind") or "").strip().lower()

    def _select_qwen3_render_skill_lines(
        self,
        context: str,
        skill_lines: List[str],
        *,
        include_skill_lines: bool,
        compact_layout: bool = False,
    ) -> List[str]:
        if not include_skill_lines or not skill_lines:
            return []
        # Keep trust skill lines visible for slot-style families where they
        # materially help multi-object and location-heavy reasoning. For state
        # families we now keep them as well, but pair them with a richer
        # natural-language snapshot so Qwen3 does not need to rely on raw
        # "[SKILL] key=value" lines alone.
        return list(skill_lines)

    def _build_qwen3_history_single_image_layout(
        self,
        context: str,
        current_step: Optional[int] = None,
        include_skill_lines: bool = True,
        compact_layout: bool = False,
        preserve_input_order: bool = False,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Build the Qwen3 structured single-image history layout once, then let
        callers choose whether to render it as a monolithic page or as cached
        structured segments.
        """
        skill_lines, blocks = self._split_qwen3_history_blocks(
            context,
            current_step=current_step,
            compact_layout=compact_layout,
        )
        rendered_skill_lines = self._select_qwen3_render_skill_lines(
            context,
            skill_lines,
            include_skill_lines=include_skill_lines,
            compact_layout=compact_layout,
        )
        snapshot_lines = self._build_qwen3_stable_snapshot_lines(
            context,
            include_skill_lines=bool(rendered_skill_lines),
            compact_layout=compact_layout,
        )
        newest_first = self._qwen3_history_blocks_newest_first(blocks)
        history_order_label = "NEWEST FIRST" if newest_first else "CHRONOLOGICAL"
        prefix_lines = self._qwen3_page_prefix_lines(
            "[RECENT HISTORY]",
            page_index=0,
            snapshot_lines=snapshot_lines,
            compact_layout=compact_layout,
            history_order_label=history_order_label,
        )
        if rendered_skill_lines:
            if prefix_lines and prefix_lines[-1] != "":
                prefix_lines.append("")
            if not compact_layout:
                prefix_lines.append("[MEMORY FACTS]")
            prefix_lines.extend(rendered_skill_lines)

        page_lines = list(prefix_lines)
        if page_lines and page_lines[-1] != "":
            page_lines.append("")

        cache_segments: List[Dict[str, str]] = []
        header_segment = self._build_qwen3_cache_segment("[RECENT HISTORY]", block_name="header/recent_history")
        if header_segment is not None:
            cache_segments.append(header_segment)
        for snapshot_line in snapshot_lines:
            snapshot_segment = self._build_qwen3_cache_segment(
                snapshot_line,
                block_name=self._qwen3_snapshot_cache_block_name(snapshot_line),
            )
            if snapshot_segment is not None:
                cache_segments.append(snapshot_segment)
        if not compact_layout:
            recent_marker_segment = self._build_qwen3_cache_segment(
                f"[RECENT HISTORY | {history_order_label}]",
                block_name="header/recent_history_order",
            )
            if recent_marker_segment is not None:
                cache_segments.append(recent_marker_segment)
        if rendered_skill_lines:
            if not compact_layout:
                memory_header_segment = self._build_qwen3_cache_segment(
                    "[MEMORY FACTS]",
                    block_name="header/memory_facts",
                )
                if memory_header_segment is not None:
                    cache_segments.append(memory_header_segment)
            for skill_line in rendered_skill_lines:
                skill_segment = self._build_qwen3_cache_segment(
                    skill_line,
                    block_name=self._qwen3_skill_cache_block_name(skill_line),
                )
                if skill_segment is not None:
                    cache_segments.append(skill_segment)

        ordered_blocks = list(blocks) if preserve_input_order else list(reversed(blocks))
        for block in ordered_blocks:
            page_lines.extend(block["lines"])
            if not compact_layout:
                page_lines.append("")
            block_segment = self._build_qwen3_cache_segment(
                "\n".join(block["lines"]),
                block_name="history/step_block",
            )
            if block_segment is not None:
                cache_segments.append(block_segment)

        page_text = "\n".join(page_lines).strip()
        kept_text = "\n".join(page_lines)
        latest_block = self._qwen3_latest_history_block(blocks)
        snapshot_state = self._collect_qwen3_snapshot_state(context)
        goal_slots = GoalSlots(
            target_objects=tuple(snapshot_state.get("goal_targets", ())),
            target_receptacle=str(snapshot_state.get("goal_receptacle", "") or "").strip() or None,
        )
        snapshot_overlap = self._compute_qwen3_snapshot_overlap(
            snapshot_lines,
            list(latest_block.get("lines", [])),
            goal_slots,
        )
        latest_observation = next(
            (line for line in latest_block.get("lines", []) if "Observation:" in str(line)),
            "",
        )
        last_action = next(
            (line for line in latest_block.get("lines", []) if "Action:" in str(line)),
            "",
        )

        diagnostics = {
            "skill_line_count": len(skill_lines),
            "rendered_skill_line_count": len(rendered_skill_lines),
            "source_block_count": len(blocks),
            "kept_block_count": len(blocks),
            "dropped_oldest_block_count": 0,
            "latest_block_visible": bool(latest_block.get("lines")) and latest_block["lines"][0] in kept_text,
            "last_action_visible": (last_action in kept_text) if last_action else True,
            "latest_observation_visible": (latest_observation in kept_text) if latest_observation else True,
            "page_count": 1,
            "cache_segment_count": len(cache_segments),
            "cache_segment_block_names": [segment["block_name"] for segment in cache_segments],
            **snapshot_overlap,
            "page_step_ranges": [
                {
                    "header": "[RECENT HISTORY]",
                    "first_step": ordered_blocks[0]["step"] if ordered_blocks else None,
                    "last_step": ordered_blocks[-1]["step"] if ordered_blocks else None,
                    "block_count": len(blocks),
                    "newest_first": newest_first,
                }
            ],
        }
        return page_text, cache_segments, diagnostics

    def _finalize_qwen3_structured_single_image(
        self,
        image_array: np.ndarray,
        *,
        image_width: int,
        min_image_height: int,
        image_padding: int,
        bg_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Reapply the page-level geometry expected by the Qwen3 structured
        renderer after assembling cached structured segments.
        """
        finalized = image_array
        if image_padding > 0:
            finalized = self._add_padding_to_array(finalized, image_padding, bg_color)

        target_height = max(int(min_image_height), int(finalized.shape[0]))
        target_width = max(int(image_width), int(finalized.shape[1]))
        if finalized.shape[0] == target_height and finalized.shape[1] == target_width:
            return finalized

        canvas = np.full((target_height, target_width, 3), bg_color, dtype=finalized.dtype)
        canvas[:finalized.shape[0], :finalized.shape[1]] = finalized
        return canvas

    def _count_qwen3_wrapped_lines(
        self,
        lines: List[str],
        available_width: int,
        font,
        font_size: int,
        use_precise: bool,
    ) -> int:
        total_lines = 0
        avg_char_width, _ = get_font_metrics(font, font_size)
        max_chars_per_line = max(1, int(available_width / max(avg_char_width, 1e-6)))
        for line in lines:
            text = str(line or "").strip()
            if not text:
                total_lines += 1
                continue
            if use_precise:
                wrapped = wrap_text_precise(text, available_width, font, font_size)
            else:
                wrapped = wrap_text_fast(text, max_chars_per_line)
            total_lines += max(1, len(wrapped))
        return total_lines

    def _compose_qwen3_history_pages(
        self,
        skill_lines: List[str],
        blocks: List[Dict[str, Any]],
        font,
        font_size: int,
        page_width: int,
        page_height: int,
        page_padding: int,
        page_budget: Optional[int],
        use_precise: bool,
        include_skill_lines: bool = True,
        compact_layout: bool = False,
        preserve_input_order: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not blocks:
            return [], {
                "skill_line_count": len(skill_lines),
                "source_block_count": 0,
                "kept_block_count": 0,
                "dropped_oldest_block_count": 0,
                "latest_block_visible": False,
                "last_action_visible": False,
                "latest_observation_visible": False,
                "page_count": 0,
            }

        available_width = max(32, page_width - 2 * page_padding)
        _, line_height = get_font_metrics(font, font_size)
        max_page_lines = max(4, int(max(1, page_height - 2 * page_padding) / max(line_height, 1)))
        newest_first = self._qwen3_history_blocks_newest_first(blocks)
        history_order_label = "NEWEST FIRST" if newest_first else "CHRONOLOGICAL"
        ordered_blocks = list(blocks) if preserve_input_order else list(reversed(blocks))
        block_line_costs = [
            self._count_qwen3_wrapped_lines(block["lines"], available_width, font, font_size, use_precise) + 1
            for block in ordered_blocks
        ]
        snapshot = self._summarize_qwen3_history_snapshot(blocks)

        pages: List[Dict[str, Any]] = []
        cursor = 0
        while cursor < len(ordered_blocks) and (page_budget is None or len(pages) < page_budget):
            page_index = len(pages)
            header = "[RECENT HISTORY]" if page_index == 0 else f"[OLDER HISTORY {page_index}]"
            prefix_lines = self._qwen3_page_prefix_lines(
                header,
                page_index=page_index,
                snapshot_lines=snapshot,
                compact_layout=compact_layout,
                history_order_label=history_order_label,
            )
            if page_index == 0 and include_skill_lines and skill_lines:
                if prefix_lines and prefix_lines[-1] != "":
                    prefix_lines.append("")
                if not compact_layout:
                    prefix_lines.append("[MEMORY FACTS]")
                prefix_lines.extend(skill_lines)
            header_line_cost = self._count_qwen3_wrapped_lines(
                prefix_lines,
                available_width,
                font,
                font_size,
                use_precise,
            ) + 1

            page_blocks: List[Dict[str, Any]] = []
            used_lines = header_line_cost
            while cursor < len(ordered_blocks):
                block_cost = block_line_costs[cursor]
                remaining_capacity = max_page_lines - used_lines
                if page_blocks and block_cost > remaining_capacity:
                    break
                page_blocks.append(ordered_blocks[cursor])
                used_lines += min(block_cost, max(1, remaining_capacity))
                cursor += 1
                if used_lines >= max_page_lines:
                    break
            if not page_blocks and cursor < len(ordered_blocks):
                page_blocks.append(ordered_blocks[cursor])
                cursor += 1
            pages.append(
                {
                    "header": header,
                    "prefix_lines": prefix_lines,
                    "blocks": page_blocks,
                }
            )

        kept_blocks = [block for page in pages for block in page["blocks"]]
        kept_text = "\n".join(
            line
            for page in pages
            for line in page.get("prefix_lines", [])
            if line
        )
        kept_text = "\n".join(
            [kept_text]
            + [
                line
                for page in pages
                for block in page["blocks"]
                for line in block["lines"]
            ]
        ).strip()
        latest_block = self._qwen3_latest_history_block(blocks)
        latest_observation = next(
            (line for line in latest_block["lines"] if "Observation:" in str(line)),
            "",
        )
        last_action = next(
            (line for line in latest_block["lines"] if "Action:" in str(line)),
            "",
        )

        diagnostics = {
            "skill_line_count": len(skill_lines),
            "source_block_count": len(blocks),
            "kept_block_count": len(kept_blocks),
            "dropped_oldest_block_count": max(0, len(blocks) - len(kept_blocks)),
            "latest_block_visible": latest_block["lines"][0] in kept_text,
            "last_action_visible": (last_action in kept_text) if last_action else True,
            "latest_observation_visible": (latest_observation in kept_text) if latest_observation else True,
            "page_count": len(pages),
            "page_step_ranges": [
                {
                    "header": page["header"],
                    "first_step": page["blocks"][0]["step"] if page["blocks"] else None,
                    "last_step": page["blocks"][-1]["step"] if page["blocks"] else None,
                    "block_count": len(page["blocks"]),
                    "newest_first": newest_first,
                }
                for page in pages
            ],
        }
        return pages, diagnostics

    def _render_qwen3_history_pages(
        self,
        trajectory_contexts: List[str],
        current_steps: Optional[List[int]],
        compact_layout_flags: Optional[List[bool]] = None,
        **override_kwargs,
    ) -> List[List[np.ndarray]]:
        config = self._get_config(**override_kwargs)
        extra_kwargs = config["extra_kwargs"]
        page_width = max(192, int(extra_kwargs.get("qwen3_history_page_width", 448)))
        page_height = max(224, int(extra_kwargs.get("qwen3_history_page_height", 512)))
        page_padding = max(6, int(extra_kwargs.get("qwen3_history_page_padding", max(8, config["padding"]))))
        page_gap = max(4, int(extra_kwargs.get("qwen3_history_page_gap", max(12, config["padding"]))))
        raw_page_budget = extra_kwargs.get("qwen3_history_page_budget", 0)
        page_budget = int(raw_page_budget or 0)
        if page_budget <= 0:
            page_budget = None
        page_columns = max(1, min(2, int(extra_kwargs.get("qwen3_history_page_columns", 2))))
        include_skill_lines = bool(extra_kwargs.get("qwen3_history_include_skill_lines", True))
        default_compact_layout = bool(extra_kwargs.get("qwen3_history_compact_layout", False))
        preserve_input_order = bool(extra_kwargs.get("qwen3_history_preserve_input_order", False))
        bg_color = config["bg_color"]
        font = _get_cached_font(config["font_path"], config["font_size"])
        diagnostics: List[Dict[str, Any]] = []
        rendered_arrays: List[List[np.ndarray]] = []

        for index, context in enumerate(trajectory_contexts):
            current_step = current_steps[index] if current_steps is not None and index < len(current_steps) else None
            compact_layout = (
                bool(compact_layout_flags[index])
                if compact_layout_flags is not None and index < len(compact_layout_flags)
                else default_compact_layout
            )
            skill_lines, blocks = self._split_qwen3_history_blocks(
                context,
                current_step=current_step,
                compact_layout=compact_layout,
            )
            pages, page_diagnostics = self._compose_qwen3_history_pages(
                skill_lines=skill_lines,
                blocks=blocks,
                font=font,
                font_size=config["font_size"],
                page_width=page_width,
                page_height=page_height,
                page_padding=page_padding,
                page_budget=page_budget,
                use_precise=config["use_precise"],
                include_skill_lines=include_skill_lines,
                compact_layout=compact_layout,
                preserve_input_order=preserve_input_order,
            )

            if not pages:
                blank = self._get_blank_array(
                    min_width=page_width,
                    min_height=page_height,
                    bg_color=bg_color,
                )
                rendered_arrays.append([blank])
                diagnostics.append(page_diagnostics)
                continue

            rendered_pages: List[Image.Image] = []
            for page in pages:
                page_lines = list(page.get("prefix_lines") or [page["header"]])
                if page_lines and page_lines[-1] != "":
                    page_lines.append("")
                for block in page["blocks"]:
                    page_lines.extend(block["lines"])
                    if not compact_layout:
                        page_lines.append("")
                page_text = "\n".join(page_lines).strip()
                page_img = trajectory_to_image(
                    page_text,
                    font_size=config["font_size"],
                    padding=page_padding,
                    bg_color=bg_color,
                    text_color=config["text_color"],
                    font_path=config["font_path"],
                    min_width=page_width,
                    max_width=page_width,
                    min_height=page_height,
                    max_height=page_height,
                    use_precise=config["use_precise"],
                    fast_mode=True,
                    compact_mode=False,
                    highlight_configs=config["highlight_configs"],
                )
                if page_img.size != (page_width, page_height):
                    fixed_page = Image.new("RGB", (page_width, page_height), bg_color)
                    fixed_page.paste(page_img, (0, 0))
                    page_img = fixed_page
                rendered_pages.append(page_img)

            page_diagnostics.update(
                {
                    "page_width": page_width,
                    "page_height": page_height,
                    "page_gap": page_gap,
                    "page_columns": min(page_columns, len(rendered_pages)),
                    "page_rows": int(math.ceil(len(rendered_pages) / max(1, min(page_columns, len(rendered_pages))))),
                    "page_layout": "multi_image",
                }
            )
            rendered_arrays.append([np.array(page_img) for page_img in rendered_pages])
            diagnostics.append(page_diagnostics)

        self._last_qwen3_history_layout_diagnostics = diagnostics
        return rendered_arrays

    def _render_qwen3_history_single_images(
        self,
        trajectory_contexts: List[str],
        current_steps: Optional[List[int]],
        **override_kwargs,
    ) -> List[np.ndarray]:
        config = self._get_config(**override_kwargs)
        extra_kwargs = config["extra_kwargs"]
        image_width = max(192, int(extra_kwargs.get("qwen3_history_page_width", config["max_width"])))
        dynamic_min_height = bool(extra_kwargs.get("qwen3_history_dynamic_min_height", False))
        if dynamic_min_height:
            min_image_height = max(0, int(extra_kwargs.get("qwen3_history_min_height", 0) or 0))
        else:
            min_image_height = max(96, int(extra_kwargs.get("qwen3_history_page_height", config["min_height"] or 96)))
        image_padding = max(6, int(extra_kwargs.get("qwen3_history_page_padding", max(8, config["padding"]))))
        include_skill_lines = bool(extra_kwargs.get("qwen3_history_include_skill_lines", True))
        default_compact_layout = bool(extra_kwargs.get("qwen3_history_compact_layout", False))
        compact_layout_flags = extra_kwargs.get("qwen3_history_compact_layout_flags")
        preserve_input_order = bool(extra_kwargs.get("qwen3_history_preserve_input_order", False))
        bg_color = config["bg_color"]
        diagnostics: List[Dict[str, Any]] = []
        rendered_arrays: List[np.ndarray] = []

        for index, context in enumerate(trajectory_contexts):
            current_step = current_steps[index] if current_steps is not None and index < len(current_steps) else None
            compact_layout = (
                bool(compact_layout_flags[index])
                if isinstance(compact_layout_flags, list) and index < len(compact_layout_flags)
                else default_compact_layout
            )
            page_text, _, layout_diagnostics = self._build_qwen3_history_single_image_layout(
                context,
                current_step=current_step,
                include_skill_lines=include_skill_lines,
                compact_layout=compact_layout,
                preserve_input_order=preserve_input_order,
            )

            page_img = trajectory_to_image(
                page_text,
                font_size=config["font_size"],
                padding=image_padding,
                bg_color=bg_color,
                text_color=config["text_color"],
                font_path=config["font_path"],
                min_width=image_width,
                max_width=image_width,
                min_height=min_image_height,
                max_height=extra_kwargs.get("max_height", config["max_height"]),
                use_precise=config["use_precise"],
                fast_mode=True,
                compact_mode=False,
                highlight_configs=config["highlight_configs"],
                )
            rendered_arrays.append(np.array(page_img))
            layout_diagnostics.update(
                {
                    "page_width": page_img.width,
                    "page_height": page_img.height,
                    "page_layout": "single_image_structured",
                }
            )
            diagnostics.append(layout_diagnostics)

        self._last_qwen3_history_layout_diagnostics = diagnostics
        return rendered_arrays
    
    def convert_texts_to_images(
        self,
        trajectory_contexts: Optional[List[str]],
        batch_size: Optional[int] = None,
        active_masks: Optional[List[bool]] = None,
        save_img: bool = False,
        compression_factor: Optional[List[float]] = None,
        resample_method: int = Image.LANCZOS,
        current_steps: Optional[List[int]] = None,
        enable_cache: bool = True,
        **override_kwargs
    ) -> List[Union[np.ndarray, List[np.ndarray]]]:
        """
        Unified method to convert trajectory texts to images or create blank images if no history.
        
        Args:
            trajectory_contexts: List of trajectory text strings (from memory.fetch()), or None/empty for blank images
            batch_size: Number of images to create (required if trajectory_contexts is None/empty)
            active_masks: List of boolean masks indicating which trajectories are active. If False, renders blank image.
            save_img: Whether to save the generated images to disk
            compression_factor: List of compression factors (one per image, should be >= 1.0). If None, no compression applied.
            resample_method: PIL resampling method for compression (default: Image.LANCZOS for best quality)
            current_steps: List of current step numbers for each environment (for incremental rendering)
            enable_cache: Enable cache-based rendering mode (requires current_steps)
            **override_kwargs: Parameters to override default configuration (can include 'step_info', 'env_idx' for custom filenames)
        
        Returns:
            List of numpy arrays representing the images
        """
        if not self.is_enabled():
            if batch_size is not None:
                return np.array([]).reshape(0, *self._get_blank_image_shape(**override_kwargs))
            return np.array([])

        qwen3_history_pages = bool(override_kwargs.get("qwen3_history_pages", False))
        qwen3_history_structured = bool(override_kwargs.get("qwen3_history_structured", False))
        self._last_qwen3_history_layout_diagnostics = []

        trust_policy_enabled = bool(override_kwargs.pop('trust_policy', False))
        trust_policy_current_steps = override_kwargs.pop('trust_policy_current_steps', current_steps)
        trust_policy_obj = override_kwargs.pop('trust_policy_obj', None)
        trust_policy_segments = override_kwargs.pop('trust_policy_segments', None)
        trust_policy_query_texts = override_kwargs.pop('trust_policy_query_texts', None)
        trust_policy_skill_feedbacks = override_kwargs.pop('trust_policy_skill_feedbacks', None)
        trust_policy_state_aware = bool(override_kwargs.pop('trust_policy_state_aware', False))
        trust_policy_context_mode = override_kwargs.pop('trust_policy_context_mode', None)
        trust_policy_collect_diagnostics = bool(override_kwargs.pop('trust_policy_collect_diagnostics', True))
        trust_policy_use_compressed_history = bool(override_kwargs.pop('trust_policy_use_compressed_history', True))
        trust_policy_use_prompt_summary = bool(override_kwargs.pop('trust_policy_use_prompt_summary', False))
        trust_policy_min_compaction_lines = max(0, int(override_kwargs.pop('trust_policy_min_compaction_lines', 0) or 0))
        trust_policy_min_prompt_summary_lines = max(
            0,
            int(override_kwargs.pop('trust_policy_min_prompt_summary_lines', trust_policy_min_compaction_lines) or 0),
        )
        trust_policy_min_history_steps_for_compaction = max(
            0,
            int(override_kwargs.pop('trust_policy_min_history_steps_for_compaction', 0) or 0),
        )
        qwen3_history_compact_layout_flags = override_kwargs.pop("qwen3_history_compact_layout_flags", None)
        qwen3_history_include_skill_lines = override_kwargs.pop("qwen3_history_include_skill_lines", None)
        if qwen3_history_include_skill_lines is None:
            # Qwen3 structured OCR splits `[SKILL]` lines from raw history.
            # When prompt summaries are enabled, move synthesized `[SKILL]`
            # facts to text and keep the OCR image focused on raw witnesses.
            if trust_policy_enabled and trust_policy_use_compressed_history:
                qwen3_history_include_skill_lines = not trust_policy_use_prompt_summary
            else:
                qwen3_history_include_skill_lines = not trust_policy_enabled
        qwen3_history_compact_layout = override_kwargs.pop("qwen3_history_compact_layout", None)
        if qwen3_history_compact_layout is None:
            qwen3_history_compact_layout = bool(trust_policy_enabled and trust_policy_use_compressed_history)
        # Rendering happens without padding and without enforced min height;
        # padding is applied only after optional compression.
        render_kwargs = {
            **override_kwargs,
            'padding': 0,
            'min_height': 0,
            'qwen3_history_include_skill_lines': bool(qwen3_history_include_skill_lines),
            'qwen3_history_compact_layout': bool(qwen3_history_compact_layout),
        }
        self._last_trust_policy_processed_contexts = []
        self._last_trust_policy_prompt_summaries = []
        blank_width = override_kwargs.get('min_width', self.min_width)
        blank_height = override_kwargs.get('min_height', self.min_height)
        if qwen3_history_pages or qwen3_history_structured:
            blank_width = max(blank_width, int(override_kwargs.get("qwen3_history_page_width", blank_width)))
            blank_height = max(blank_height, int(override_kwargs.get("qwen3_history_page_height", blank_height)))
        
        # If no trajectory contexts provided, create blank images
        if trajectory_contexts is None or len(trajectory_contexts) == 0:
            if batch_size is None:
                raise ValueError("batch_size must be provided when trajectory_contexts is None or empty")
            blank_kwargs = {**override_kwargs, "min_width": blank_width, "min_height": blank_height}
            image_arrays = self.create_blank_images(batch_size, **blank_kwargs)
            if qwen3_history_pages:
                image_arrays = [[img] for img in image_arrays]
            if qwen3_history_pages or qwen3_history_structured:
                self._last_qwen3_history_layout_diagnostics = [
                    {"blank": True, "page_count": 0}
                    for _ in range(batch_size)
                ]
        else:
            trajectory_contexts = preprocess_trajectory_contexts(trajectory_contexts)
            if trust_policy_enabled:
                trajectory_contexts, trust_compression_factor = self._apply_trust_policy_to_contexts(
                    trajectory_contexts,
                    current_steps=trust_policy_current_steps,
                    trust_policy_obj=trust_policy_obj,
                    trust_policy_segments=trust_policy_segments,
                    trust_policy_query_texts=trust_policy_query_texts,
                    trust_policy_skill_feedbacks=trust_policy_skill_feedbacks,
                    trust_policy_state_aware=trust_policy_state_aware,
                    trust_policy_context_mode=trust_policy_context_mode,
                    collect_diagnostics=trust_policy_collect_diagnostics,
                    use_compressed_history=trust_policy_use_compressed_history,
                    build_prompt_summary=trust_policy_use_prompt_summary,
                    min_compaction_lines=trust_policy_min_compaction_lines,
                    min_prompt_summary_lines=trust_policy_min_prompt_summary_lines,
                    min_history_steps_for_compaction=trust_policy_min_history_steps_for_compaction,
                )
                self._last_trust_policy_processed_contexts = list(trajectory_contexts)
                qwen3_history_compact_layout_flags = [
                    bool((diagnostics or {}).get("trust_policy/render_compacted", 0.0) > 0.0)
                    for diagnostics in self._last_trust_policy_diagnostics
                ]
                if compression_factor is None:
                    compression_factor = trust_compression_factor
                else:
                    compression_factor = [
                        min(float(base_cf), float(trust_cf))
                        for base_cf, trust_cf in zip(compression_factor, trust_compression_factor)
                    ]
            # If active_masks is None, set all to True
            if active_masks is None:
                active_masks = [True] * len(trajectory_contexts)
            
            if len(active_masks) != len(trajectory_contexts):
                raise ValueError(f"Length of active_masks ({len(active_masks)}) must match length of trajectory_contexts ({len(trajectory_contexts)})")
            
            # Pre-create blank array for inactive entries
            bg_color = override_kwargs.get('bg_color', self.bg_color)
            blank_img = Image.new('RGB', (blank_width, blank_height), bg_color)
            blank_array = np.array(blank_img)
            
            # Separate active and inactive indices
            active_indices = [i for i, mask in enumerate(active_masks) if mask]
            inactive_indices = [i for i, mask in enumerate(active_masks) if not mask]
            full_qwen3_layout_diagnostics: List[Dict[str, Any]] = [
                {"inactive": True, "page_count": 0} for _ in trajectory_contexts
            ]
            
            # Only process active trajectories
            if active_indices:
                active_contexts = [trajectory_contexts[i] for i in active_indices]
                active_current_steps = [current_steps[i] for i in active_indices] if current_steps is not None else None
                active_compact_layout_flags = (
                    [bool(qwen3_history_compact_layout_flags[i]) for i in active_indices]
                    if isinstance(qwen3_history_compact_layout_flags, list)
                    else None
                )
                
                if qwen3_history_pages:
                    active_image_arrays = self._render_qwen3_history_pages(
                        active_contexts,
                        active_current_steps,
                        compact_layout_flags=active_compact_layout_flags,
                        **render_kwargs,
                    )
                    active_qwen3_diagnostics = list(self._last_qwen3_history_layout_diagnostics)
                elif qwen3_history_structured:
                    qwen3_structured_config = self._get_config(**render_kwargs)
                    qwen3_structured_extra = qwen3_structured_config["extra_kwargs"]
                    image_width = max(192, int(qwen3_structured_extra.get("qwen3_history_page_width", qwen3_structured_config["max_width"])))
                    dynamic_min_height = bool(qwen3_structured_extra.get("qwen3_history_dynamic_min_height", False))
                    if dynamic_min_height:
                        min_image_height = max(0, int(qwen3_structured_extra.get("qwen3_history_min_height", 0) or 0))
                    else:
                        min_image_height = max(96, int(qwen3_structured_extra.get("qwen3_history_page_height", qwen3_structured_config["min_height"] or 96)))
                    image_padding = max(6, int(qwen3_structured_extra.get("qwen3_history_page_padding", max(8, qwen3_structured_config["padding"]))))
                    content_width = max(32, image_width - 2 * image_padding)
                    structured_segments: List[List[str]] = []
                    active_qwen3_diagnostics = []
                    for context_index, context in enumerate(active_contexts):
                        current_step = (
                            active_current_steps[context_index]
                            if active_current_steps is not None and context_index < len(active_current_steps)
                            else None
                        )
                        compact_layout = (
                            bool(active_compact_layout_flags[context_index])
                            if active_compact_layout_flags is not None and context_index < len(active_compact_layout_flags)
                            else bool(qwen3_structured_extra.get("qwen3_history_compact_layout", False))
                        )
                        _, cache_segments, layout_diagnostics = self._build_qwen3_history_single_image_layout(
                            context,
                            current_step=current_step,
                            include_skill_lines=bool(qwen3_structured_extra.get("qwen3_history_include_skill_lines", True)),
                            compact_layout=compact_layout,
                        )
                        structured_segments.append(cache_segments)
                        active_qwen3_diagnostics.append(layout_diagnostics)

                    if enable_cache and active_current_steps is not None and self.enable_cache:
                        structured_render_kwargs = {
                            **render_kwargs,
                            "min_width": content_width,
                            "max_width": content_width,
                            "min_height": 0,
                            "padding": 0,
                            "compact_mode": False,
                        }
                        active_image_arrays = self._convert_incremental_segments(
                            structured_segments,
                            env_indices=active_indices,
                            batch_size=len(active_contexts),
                            **structured_render_kwargs,
                        )
                        finalized_arrays: List[np.ndarray] = []
                        for layout_diagnostics, image_array in zip(active_qwen3_diagnostics, active_image_arrays):
                            finalized_array = self._finalize_qwen3_structured_single_image(
                                image_array,
                                image_width=image_width,
                                min_image_height=min_image_height,
                                image_padding=image_padding,
                                bg_color=qwen3_structured_config["bg_color"],
                            )
                            layout_diagnostics.update(
                                {
                                    "page_width": int(finalized_array.shape[1]),
                                    "page_height": int(finalized_array.shape[0]),
                                    "page_layout": "single_image_structured_cached",
                                }
                            )
                            finalized_arrays.append(finalized_array)
                        active_image_arrays = finalized_arrays
                    else:
                        active_image_arrays = self._render_qwen3_history_single_images(
                            active_contexts,
                            active_current_steps,
                            qwen3_history_compact_layout_flags=active_compact_layout_flags,
                            **render_kwargs,
                        )
                        active_qwen3_diagnostics = list(self._last_qwen3_history_layout_diagnostics)
                # Incremental rendering mode for active trajectories
                elif enable_cache and active_current_steps is not None and self.enable_cache:
                    compact_mode = override_kwargs.get('compact_mode', self.compact_mode)
                    if compact_mode:
                        active_image_arrays = self._convert_incremental_compact(
                            active_contexts, 
                            active_current_steps, 
                            env_indices=active_indices,
                            batch_size=batch_size,
                            **render_kwargs
                        )
                    else:
                        active_image_arrays = self._convert_incremental(
                            active_contexts, 
                            active_current_steps,
                            env_indices=active_indices,
                            batch_size=batch_size,
                            **render_kwargs
                        )
                else:
                    # Normal rendering mode for active trajectories
                    active_images = self.convert_batch(active_contexts, **render_kwargs)
                    active_image_arrays = []
                    for img in active_images:
                        if img is not None:
                            active_image_arrays.append(np.array(img))
                        else:
                            active_image_arrays.append(blank_array.copy())
                    active_qwen3_diagnostics = []
            else:
                active_image_arrays = []
                active_qwen3_diagnostics = []
            
            # Reconstruct full array with blanks for inactive entries
            image_arrays = [None] * len(trajectory_contexts)
            for idx, img_array in zip(active_indices, active_image_arrays):
                image_arrays[idx] = img_array
            for idx in inactive_indices:
                image_arrays[idx] = [blank_array.copy()] if qwen3_history_pages else blank_array.copy()
            if qwen3_history_pages or qwen3_history_structured:
                for idx, layout_diag in zip(active_indices, active_qwen3_diagnostics):
                    full_qwen3_layout_diagnostics[idx] = layout_diag
                self._last_qwen3_history_layout_diagnostics = full_qwen3_layout_diagnostics
        
        # Apply compression if specified
        if compression_factor is not None:
            if len(compression_factor) != len(image_arrays):
                raise ValueError(f"Length of compression_factor ({len(compression_factor)}) must match length of image_arrays ({len(image_arrays)})")
            invalid_factors = [cf for cf in compression_factor if cf < 1.0]
            if invalid_factors:
                raise ValueError(f"All compression_factors must be >= 1.0, got {invalid_factors}")
            # Only compress if at least one factor > 1.0 (compress_image_arrays handles cf == 1.0 by skipping)
            if any(cf > 1.0 for cf in compression_factor):
                if qwen3_history_pages:
                    compressed_arrays: List[Union[np.ndarray, List[np.ndarray]]] = []
                    for pages, cf in zip(image_arrays, compression_factor):
                        if isinstance(pages, list):
                            compressed_arrays.append(
                                self.compress_image_arrays(
                                    pages,
                                    [cf] * len(pages),
                                    resample_method=resample_method,
                                )
                            )
                        else:
                            compressed_arrays.append(
                                self.compress_image_arrays(
                                    [pages],
                                    [cf],
                                    resample_method=resample_method,
                                )[0]
                            )
                    image_arrays = compressed_arrays
                else:
                    image_arrays = self.compress_image_arrays(
                        image_arrays,
                        compression_factor=compression_factor,
                        resample_method=resample_method
                    )
        
        # Apply padding after compression so that borders are not compressed.
        padding_to_add = override_kwargs.get('padding', self.padding)
        if padding_to_add and padding_to_add > 0:
            bg_color = override_kwargs.get('bg_color', self.bg_color)
            padded_arrays: List[Union[np.ndarray, List[np.ndarray]]] = []
            for arr in image_arrays:
                if isinstance(arr, list):
                    padded_arrays.append(
                        [self._add_padding_to_array(page, padding_to_add, bg_color) for page in arr]
                    )
                else:
                    padded_arrays.append(self._add_padding_to_array(arr, padding_to_add, bg_color))
            image_arrays = padded_arrays

        if trust_policy_enabled:
            image_arrays = self._attach_trust_policy_mm_metadata(
                image_arrays,
                qwen3_history_pages=qwen3_history_pages,
            )
        
        # Save images if requested (save after compression to save disk space)
        if save_img and image_arrays:
            self._save_images(image_arrays, **override_kwargs)

        if compression_factor is None:
            self._last_applied_compression_factors = [1.0] * len(image_arrays)
        else:
            self._last_applied_compression_factors = [float(cf) for cf in compression_factor]
        
        return image_arrays

    def _apply_trust_policy_to_single_context(
        self,
        *,
        context: str,
        current_step: int,
        policy: TrustCalibratedRenderPolicy,
        structured_segments: Optional[List[Dict[str, Any]]] = None,
        query_text: Optional[str] = None,
        skill_feedback: Optional[MemorySkillFeedback] = None,
        trust_policy_state_aware: bool = False,
        trust_policy_context_mode: Optional[str] = None,
        collect_diagnostics: bool = True,
        use_compressed_history: bool = True,
        build_prompt_summary: bool = False,
        min_compaction_lines: int = 0,
        min_prompt_summary_lines: int = 0,
        min_history_steps_for_compaction: int = 0,
    ) -> Tuple[str, float, Dict[str, Any], str]:
        raw_lines = [line for line in context.split('\n') if line.strip()]
        prepared_context: Optional[PreparedTrustContext] = None
        if query_text:
            prepared_context = prepare_trust_context(
                query_text=query_text,
                raw_lines=raw_lines,
                requested_mode=trust_policy_context_mode,
                state_aware=trust_policy_state_aware,
                feedback=skill_feedback,
            )

        segments = self._build_trust_segments(
            context=context,
            current_step=current_step,
            structured_segments=structured_segments,
            query_text=query_text,
            skill_feedback=skill_feedback,
            state_aware=trust_policy_state_aware,
            context_mode=trust_policy_context_mode,
            prepared_context=prepared_context,
        )

        decisions = policy.decide_batch(segments, current_step=current_step)
        if use_compressed_history and query_text:
            decisions = policy.apply_context_budget(decisions)
        effective_min_compaction_lines = max(0, int(min_compaction_lines))
        effective_min_history_steps_for_compaction = max(0, int(min_history_steps_for_compaction))
        if prepared_context is not None:
            effective_min_compaction_lines = _phase_aware_min_compaction_lines(
                effective_min_compaction_lines,
                family=prepared_context.feedback_family,
                phase=prepared_context.feedback_phase,
            )
        compaction_gate_delayed = bool(
            use_compressed_history
            and query_text
            and prepared_context is not None
            and prepared_context.resolved_mode != "query"
            and len(raw_lines) < effective_min_compaction_lines
        )
        step_compaction_gate_active = bool(
            effective_min_history_steps_for_compaction > 0
            and current_step < effective_min_history_steps_for_compaction
        )
        render_skill_lines_in_raw = bool(
            use_compressed_history
            or prepared_context is None
            or prepared_context.resolved_mode == "query"
        )
        rendered_lines = [
            decision.text
            for segment, decision in zip(segments, decisions)
            if (
                decision.action != "hide"
                and (use_compressed_history or render_skill_lines_in_raw or decision.action != "warn_low_res")
                and str(decision.text or "").strip()
                and (
                    render_skill_lines_in_raw
                    or not str(segment.text or "").strip().startswith("[SKILL]")
                )
            )
        ]
        visible_raw_lines = [
            str(segment.text).strip()
            for segment, decision in zip(segments, decisions)
            if (
                decision.action != "hide"
                and (use_compressed_history or render_skill_lines_in_raw or decision.action != "warn_low_res")
                and str(segment.text or "").strip()
                and (
                    render_skill_lines_in_raw
                    or not str(segment.text or "").strip().startswith("[SKILL]")
                )
            )
        ]
        visible_history_lines = [
            line
            for line in visible_raw_lines
            if line and not str(line).strip().startswith("[SKILL]")
        ]
        renderable_text = "\n".join(rendered_lines)
        if (
            use_compressed_history
            and query_text
            and not compaction_gate_delayed
            and not step_compaction_gate_active
        ):
            rendered_line_total = len([line for line in renderable_text.split('\n') if line.strip()])
            rendered_char_total = sum(len(str(line)) for line in rendered_lines)
            source_lines_for_budget = raw_lines
            source_line_total = len(source_lines_for_budget)
            source_char_total = sum(len(str(line)) for line in source_lines_for_budget)
            expanded_vs_source = (
                rendered_line_total > source_line_total
                or rendered_char_total > source_char_total
            )
            should_try_compact = (
                len(raw_lines) >= effective_min_compaction_lines
                or expanded_vs_source
            )
            if should_try_compact:
                compact_kept_lines = visible_raw_lines
                if prepared_context is not None and prepared_context.resolved_mode == "query":
                    compact_kept_lines = [
                        str(segment.text).strip()
                        for segment, decision in zip(segments, decisions)
                        if decision.action == "full_res" and str(segment.text or "").strip()
                    ]
                    if not compact_kept_lines:
                        compact_kept_lines = visible_raw_lines
                compact_text = build_compact_trust_context(
                    query_text=query_text,
                    raw_lines=raw_lines,
                    kept_lines=compact_kept_lines,
                    requested_mode=trust_policy_context_mode,
                    state_aware=trust_policy_state_aware,
                    feedback=skill_feedback,
                    context_budget_percent=policy.config.context_budget_percent,
                    prompt_summary_active=build_prompt_summary,
                    prepared_context=prepared_context,
                )
                if compact_text:
                    compact_lines = [line for line in compact_text.split('\n') if line.strip()]
                    compact_line_total = len(compact_lines)
                    compact_char_total = sum(len(str(line)) for line in compact_lines)
                    structured_compact = any(line.startswith("[SKILL]") for line in compact_lines)
                    allow_structured_expansion = (
                        structured_compact
                        and compact_line_total <= max(4, source_line_total + 2)
                        and compact_char_total <= max(int(source_char_total * 3.5), source_char_total + 220)
                    )
                    if (
                        compact_line_total <= source_line_total
                        and compact_char_total <= source_char_total
                    ) or allow_structured_expansion:
                        renderable_text = compact_text
                    elif (
                        compact_line_total < source_line_total
                        and compact_char_total <= max(int(source_char_total * 1.25), source_char_total + 64)
                    ):
                        renderable_text = compact_text
                    elif expanded_vs_source:
                        renderable_text = "\n".join(source_lines_for_budget)
                elif expanded_vs_source:
                    renderable_text = "\n".join(source_lines_for_budget)
            elif expanded_vs_source:
                renderable_text = "\n".join(source_lines_for_budget)
        if compaction_gate_delayed:
            renderable_text = "\n".join(raw_lines)
        fallback_recent_history_used = 0.0
        fallback_recent_history_line_count = 0.0
        if not renderable_text and raw_lines:
            recent_history_lines: List[str] = []
            last_step_header_index = None
            for idx in range(len(raw_lines) - 1, -1, -1):
                if str(raw_lines[idx]).strip().startswith("[Step "):
                    last_step_header_index = idx
                    break
            if last_step_header_index is not None:
                recent_history_lines = raw_lines[last_step_header_index:]
            if not recent_history_lines:
                recent_history_lines = raw_lines[-min(6, len(raw_lines)):]
            renderable_text = "\n".join(recent_history_lines)
            fallback_recent_history_used = 1.0 if renderable_text else 0.0
            fallback_recent_history_line_count = float(len(recent_history_lines))
        if not renderable_text:
            renderable_text = ""

        summary_text = ""
        if build_prompt_summary and query_text and len(raw_lines) >= max(0, int(min_prompt_summary_lines)):
            summary_text = build_trust_policy_text_summary(
                query_text=query_text,
                raw_lines=raw_lines,
                requested_mode=trust_policy_context_mode,
                state_aware=trust_policy_state_aware,
                feedback=skill_feedback,
                prepared_context=prepared_context,
            )

        diagnostics = collect_trust_policy_monitor(
            query_text=query_text,
            raw_lines=raw_lines,
            rendered_lines=[line for line in renderable_text.split('\n') if line.strip()],
            prompt_summary_text=summary_text,
            requested_mode=trust_policy_context_mode,
            state_aware=trust_policy_state_aware,
            detailed=collect_diagnostics,
            prepared_context=prepared_context,
        )
        diagnostics["trust_policy/fallback_recent_history_used"] = fallback_recent_history_used
        diagnostics["trust_policy/fallback_recent_history_line_count"] = fallback_recent_history_line_count
        diagnostics["trust_policy/compaction_step_gate_active"] = (
            1.0 if step_compaction_gate_active else 0.0
        )
        kept_factors = [decision.compression_factor for decision in decisions if decision.action != "hide"]
        # Segment-level trust decisions are collapsed back into a single OCR
        # image. The image-wide compression therefore must respect the most
        # sensitive retained line, not the least sensitive one.
        compression_factor = min(kept_factors) if kept_factors else 1.0
        return renderable_text, compression_factor, diagnostics, summary_text

    def _apply_trust_policy_to_contexts(
        self,
        trajectory_contexts: List[str],
        current_steps: Optional[List[int]] = None,
        trust_policy_obj: Optional[TrustCalibratedRenderPolicy] = None,
        trust_policy_segments: Optional[List[List[Dict[str, Any]]]] = None,
        trust_policy_query_texts: Optional[List[str]] = None,
        trust_policy_skill_feedbacks: Optional[List[Optional[MemorySkillFeedback]]] = None,
        trust_policy_state_aware: bool = False,
        trust_policy_context_mode: Optional[str] = None,
        collect_diagnostics: bool = True,
        use_compressed_history: bool = True,
        build_prompt_summary: bool = False,
        min_compaction_lines: int = 0,
        min_prompt_summary_lines: int = 0,
        min_history_steps_for_compaction: int = 0,
    ) -> Tuple[List[str], List[float]]:
        """
        Apply trust-calibrated optical memory policy before rendering.

        This is opt-in and intentionally conservative: by default it only
        rewrites each context into renderable trusted lines and returns a
        per-image compression suggestion derived from the least-compressed
        surviving line.
        """
        policy = trust_policy_obj or TrustCalibratedRenderPolicy()

        def _task(index: int) -> Tuple[str, float, Dict[str, Any], str]:
            context = trajectory_contexts[index]
            raw_lines = [line for line in context.split('\n') if line.strip()]
            current_step = (
                int(current_steps[index])
                if current_steps is not None and index < len(current_steps)
                else len(raw_lines)
            )
            query_text = (
                trust_policy_query_texts[index]
                if trust_policy_query_texts and index < len(trust_policy_query_texts)
                else None
            )
            skill_feedback = (
                trust_policy_skill_feedbacks[index]
                if trust_policy_skill_feedbacks and index < len(trust_policy_skill_feedbacks)
                else None
            )
            structured_segments = (
                trust_policy_segments[index]
                if trust_policy_segments and index < len(trust_policy_segments)
                else None
            )
            return self._apply_trust_policy_to_single_context(
                context=context,
                current_step=current_step,
                policy=policy,
                structured_segments=structured_segments,
                query_text=query_text,
                skill_feedback=skill_feedback,
                trust_policy_state_aware=trust_policy_state_aware,
                trust_policy_context_mode=trust_policy_context_mode,
                collect_diagnostics=collect_diagnostics,
                use_compressed_history=use_compressed_history,
                build_prompt_summary=build_prompt_summary,
                min_compaction_lines=min_compaction_lines,
                min_prompt_summary_lines=min_prompt_summary_lines,
                min_history_steps_for_compaction=min_history_steps_for_compaction,
            )

        results: List[Tuple[str, float, Dict[str, Any], str]] = []
        for index in range(len(trajectory_contexts)):
            results.append(_task(index))

        processed_contexts = [result[0] for result in results]
        compression_factors = [result[1] for result in results]
        diagnostics = [result[2] for result in results]
        prompt_summaries = [result[3] for result in results]
        self._last_trust_policy_diagnostics = diagnostics
        self._last_trust_policy_prompt_summaries = prompt_summaries
        return processed_contexts, compression_factors

    def _build_trust_segments(
        self,
        context: str,
        current_step: int,
        structured_segments: Optional[List[Dict[str, Any]]] = None,
        query_text: Optional[str] = None,
        skill_feedback: Optional[MemorySkillFeedback] = None,
        state_aware: bool = False,
        context_mode: Optional[str] = None,
        prepared_context: Optional[PreparedTrustContext] = None,
    ) -> List[SegmentTrustMetadata]:
        """Build trust-policy segments from either structured metadata or raw lines."""

        if structured_segments:
            segments = []
            for item in structured_segments:
                if isinstance(item, SegmentTrustMetadata):
                    segments.append(item)
                    continue
                segments.append(
                    SegmentTrustMetadata(
                        text=str(item.get("text", "")),
                        step=int(item.get("step", current_step)),
                        source_id=str(item.get("source_id", "env")),
                        source_trust=float(item.get("source_trust", 1.0)),
                        support_count=int(item.get("support_count", 1)),
                        contradiction_count=int(item.get("contradiction_count", 0)),
                        suspicious_score=float(item.get("suspicious_score", 0.0)),
                        salience=float(item.get("salience", 0.5)),
                        query_relevance=float(item.get("query_relevance", 0.5)),
                    )
                )
            return segments

        segments = []
        if query_text:
            return build_query_conditioned_segments_from_lines(
                context.split('\n'),
                current_step=current_step,
                query_text=query_text,
                skill_feedback=skill_feedback,
                state_aware=state_aware,
                context_mode=context_mode,
                prepared_context=prepared_context,
            )
        return build_trust_segments_from_lines(
            context.split('\n'),
            current_step=current_step,
        )
    
    def _get_blank_image_shape(self, **override_kwargs) -> Tuple[int, int, int]:
        """Get the shape of a blank image (H, W, 3)."""
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        return (height, width, 3)
    
    def _get_blank_array(self, **override_kwargs) -> np.ndarray:
        """Get a blank image as numpy array."""
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        bg_color = override_kwargs.get('bg_color', self.bg_color)
        blank_img = Image.new('RGB', (width, height), bg_color)
        return np.array(blank_img)
    
    def create_blank_images(
        self,
        batch_size: int,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Create a batch of blank images (useful for first step when there's no history).
        
        Args:
            batch_size: Number of blank images to create
            **override_kwargs: Parameters to override default configuration (e.g., min_width, min_height, bg_color)
        
        Returns:
            List of numpy arrays representing the blank images
        """
        if not self.is_enabled():
            return np.array([])
        
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        bg_color = override_kwargs.get('bg_color', self.bg_color)
        
        blank_image = Image.new('RGB', (width, height), bg_color)
        blank_array = np.array(blank_image)
        # Stack the same blank image batch_size times
        return [blank_array] * batch_size
    
    def compress_image_arrays(
        self,
        image_arrays: List[np.ndarray],
        compression_factor: List[float],
        keep_aspect_ratio: bool = True,
        resample_method: int = Image.LANCZOS
    ) -> List[np.ndarray]:
        """
        Compress image arrays by a given factor while maintaining image clarity.
        
        Uses high-quality resampling (Lanczos by default) to preserve sharpness and details
        during downscaling. This is particularly useful for reducing memory usage and 
        computational costs while keeping OCR-readable images.
        
        Args:
            image_arrays: List of numpy arrays to compress
            compression_factor: List of factors by which to compress each image (e.g., 2.0 means halving the dimensions)
                              Must be >= 1.0 (1.0 = no compression, > 1.0 = compress). One factor per image.
            keep_aspect_ratio: Whether to maintain the original aspect ratio (default: True)
            resample_method: PIL resampling method. Options:
                           - Image.LANCZOS (default): Highest quality for downsampling
                           - Image.BICUBIC: Good quality, faster than Lanczos
                           - Image.BILINEAR: Faster but lower quality
                           - Image.NEAREST: Fastest but lowest quality
        
        Returns:
            List of compressed image arrays
        
        Examples:
            >>> # Compress batch of images with different factors per image
            >>> compressed_batch = ocr_tool.compress_image_arrays(images, [1.5, 2.0, 1.0])
            
            >>> # Use faster but lower quality resampling
            >>> compressed = ocr_tool.compress_image_arrays(images, [2.0, 2.0], resample_method=Image.BICUBIC)
        """
        if len(compression_factor) != len(image_arrays):
            raise ValueError(f"Length of compression_factor ({len(compression_factor)}) must match length of image_arrays ({len(image_arrays)})")
        
        for cf in compression_factor:
            if cf < 1.0:
                raise ValueError(f"All compression_factors must be >= 1.0, got {cf}")
        
        compressed_arrays = []
        
        for img_array, cf in zip(image_arrays, compression_factor):
            if img_array is None or not isinstance(img_array, np.ndarray):
                compressed_arrays.append(img_array)
                continue
            
            # Skip compression if factor is 1.0 (no compression)
            if cf == 1.0:
                compressed_arrays.append(img_array)
                continue
            
            # Get original dimensions
            height, width = img_array.shape[:2]
            
            # Calculate new dimensions, sqrt(cf) is the factor by which the dimensions are reduced
            new_width = max(28, int(width / math.sqrt(cf)))
            new_height = max(28, int(height / math.sqrt(cf)))
            
            # Ensure minimum dimensions for readability
            new_width = max(new_width, self.min_width)
            new_height = max(new_height, self.min_height)
            
            # Convert numpy array to PIL Image
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            img = Image.fromarray(img_array)
            
            # Resize using high-quality resampling
            compressed_img = img.resize((new_width, new_height), resample=resample_method)
            
            # Convert back to numpy array
            compressed_array = np.array(compressed_img)
            compressed_arrays.append(compressed_array)
        
        return compressed_arrays
    
    def _add_padding_to_array(
        self,
        img_array: Optional[np.ndarray],
        padding: int,
        bg_color: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Add uniform padding around an image array using the given background color.
        """
        if img_array is None or not isinstance(img_array, np.ndarray) or padding <= 0:
            return img_array
        
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array)
        padded_img = ImageOps.expand(img, border=padding, fill=bg_color)
        return np.array(padded_img)
    
    def _save_images(
        self,
        image_arrays: List[np.ndarray],
        **kwargs
    ) -> None:
        """
        Save trajectory images to disk.
        
        Args:
            image_arrays: List of numpy arrays representing images
            **kwargs: Additional parameters for customizing filenames (e.g., 'step_info', 'env_idx')
        """
        from datetime import datetime
        
        step_info = kwargs.get('step_info', 'unknown')
        
        for i, img_array in enumerate(image_arrays):
            if img_array is not None:
                # Convert numpy array to PIL Image
                if isinstance(img_array, np.ndarray):
                    img = Image.fromarray(img_array.astype(np.uint8))
                else:
                    img = img_array
                
                # Create filename with optional custom info
                env_idx = kwargs.get('env_idx', i)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trajectory_env{env_idx}_{step_info}_{self.image_save_counter:06d}_{timestamp}.png"
                filepath = os.path.join(self.trajectory_images_dir, filename)
                img.save(filepath)
                # print(f"Saved trajectory image to: {filepath}")
        
        self.image_save_counter += 1
