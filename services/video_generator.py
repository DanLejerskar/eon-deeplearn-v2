"""
EON DeepLearn™ Video Generator Service
Per Technical Specification v1.0

Pipeline:
1. Voice Synthesis - Coqui XTTS v2 / ElevenLabs / gTTS fallback
2. Visual Assembly - DALL-E 3 / gradient fallback
3. Video Assembly - FFmpeg per spec
"""

import os

# Check if we can use the real implementation
try:
    from services.video_generator_real import VideoGeneratorReal
    USE_REAL_GENERATOR = True
    print("[VideoGenerator] ✓ Using PRODUCTION video generation per spec")
except ImportError as e:
    USE_REAL_GENERATOR = False
    print(f"[VideoGenerator] ✗ Falling back to MOCK mode: {e}")


class VideoGenerator:
    """
    Video Generator with automatic selection between real and mock implementations.
    Prefers real implementation when dependencies are available.
    """

    def __init__(self):
        self.output_dir = "data/outputs"
        self.temp_dir = "data/temp"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Try to use real generator per spec
        if USE_REAL_GENERATOR:
            openai_key = os.environ.get('OPENAI_API_KEY', '')
            runpod_key = os.environ.get('RUNPOD_API_KEY', '')
            runpod_endpoint = os.environ.get('RUNPOD_ENDPOINT_ID', '5ncsdkoclqv8yj')
            self._real_generator = VideoGeneratorReal(
                openai_api_key=openai_key,
                runpod_api_key=runpod_key,
                runpod_endpoint_id=runpod_endpoint
            )
            self._use_real = True
            print(f"[VideoGenerator] OpenAI API: {'configured' if openai_key else 'not set (using gradient images)'}")
            print(f"[VideoGenerator] RunPod TTS: {'configured' if runpod_key else 'not set (using gTTS fallback)'}")
        else:
            self._real_generator = None
            self._use_real = False

    def generate_voice(self, script: dict, voice_style: str, job_id: str,
                      progress_callback=None) -> str:
        """Generate voice narration from script"""
        if self._use_real and self._real_generator:
            return self._real_generator.generate_voice(script, voice_style, job_id, progress_callback)
        else:
            return self._mock_generate_voice(script, voice_style, job_id, progress_callback)

    def generate_visuals(self, script: dict, style: str, frequency: int,
                        job_id: str, progress_callback=None) -> list:
        """Generate visual assets for the video"""
        if self._use_real and self._real_generator:
            return self._real_generator.generate_visuals(script, style, frequency, job_id, progress_callback)
        else:
            return self._mock_generate_visuals(script, style, frequency, job_id, progress_callback)

    def assemble_video(self, voice_file: str, visuals: list, job_id: str,
                      progress_callback=None) -> str:
        """Assemble final video from voice and visuals"""
        if self._use_real and self._real_generator:
            return self._real_generator.assemble_video(voice_file, visuals, job_id, progress_callback)
        else:
            return self._mock_assemble_video(voice_file, visuals, job_id, progress_callback)

    # ============ MOCK IMPLEMENTATIONS (fallback) ============

    def _mock_generate_voice(self, script: dict, voice_style: str, job_id: str,
                            progress_callback=None) -> str:
        """Mock voice generation - creates placeholder file"""
        import time
        import json
        from datetime import datetime

        print(f"[MOCK] Generating voice for job {job_id}")

        total_modules = len(script.get('modules', []))
        for i in range(max(total_modules, 1)):
            if progress_callback:
                progress_callback(int((i + 1) / max(total_modules, 1) * 100))
            time.sleep(0.3)

        output_file = os.path.join(self.output_dir, f"{job_id}_voice.mp3")

        with open(output_file, 'w') as f:
            f.write(f"[MOCK AUDIO FILE]\nJob: {job_id}\nStyle: {voice_style}\n")
            f.write(f"Duration: {script.get('target_duration_hours', 1)} hours\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"\nNote: Install gTTS and Pillow for real video generation.\n")

        return output_file

    def _mock_generate_visuals(self, script: dict, style: str, frequency: int,
                              job_id: str, progress_callback=None) -> list:
        """Mock visual generation - creates placeholder list"""
        import time

        print(f"[MOCK] Generating visuals for job {job_id}")

        total_minutes = script.get('target_duration_hours', 1) * 60
        total_seconds = total_minutes * 60
        num_images = int(total_seconds // frequency)
        num_images = min(num_images, 30)

        visuals = []
        for i in range(num_images):
            if progress_callback:
                progress_callback(int((i + 1) / num_images * 100))

            image_path = os.path.join(self.output_dir, f"{job_id}_img_{i:03d}.jpg")
            visuals.append({
                "path": image_path,
                "timestamp": i * frequency,
                "description": f"[MOCK] Visual {i + 1}"
            })
            time.sleep(0.1)

        return visuals

    def _mock_assemble_video(self, voice_file: str, visuals: list, job_id: str,
                            progress_callback=None) -> str:
        """Mock video assembly - creates placeholder file"""
        import time
        import json
        from datetime import datetime

        print(f"[MOCK] Assembling video for job {job_id}")

        stages = ['Preparing', 'Processing', 'Combining', 'Finalizing']
        for i, stage in enumerate(stages):
            if progress_callback:
                progress_callback(int((i + 1) / len(stages) * 100))
            time.sleep(0.3)

        output_file = os.path.join(self.output_dir, f"{job_id}_final.mp4")

        job_info = {
            "job_id": job_id,
            "voice_file": voice_file,
            "num_visuals": len(visuals),
            "created_at": datetime.now().isoformat(),
            "status": "mock_generated",
            "note": "MOCK VIDEO - Install gTTS, Pillow, and FFmpeg for real videos."
        }

        with open(output_file, 'w') as f:
            json.dump(job_info, f, indent=2)

        return output_file
