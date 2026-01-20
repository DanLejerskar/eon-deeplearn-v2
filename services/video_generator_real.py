"""
EON DeepLearn™ Video Generator - PRODUCTION IMPLEMENTATION
Per Technical Specification v1.0

Voice Synthesis: RunPod Chatterbox TTS (production quality)
Image Generation: DALL-E 3 (or gradient fallback)
Video Assembly: FFmpeg with stillimage tune
"""

import os
import time
import json
import requests
import subprocess
import base64
import re
from datetime import datetime
from typing import Optional, Callable

# Try PIL for image generation fallback
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VideoGeneratorReal:
    """
    Production video generator implementing EON DeepLearn™ spec.

    Pipeline stages:
    1. Voice Synthesis - RunPod Chatterbox TTS (outperforms ElevenLabs!)
    2. Visual Assembly - DALL-E 3 hero images
    3. Video Assembly - FFmpeg with optimized settings
    """

    # EON Narrator Voice Profiles per spec
    VOICE_PROFILES = {
        "british_documentary": {
            "description": "Calm, authoritative British-accented voice (documentary style)",
            "exaggeration": 0.3,  # Calm
            "cfg_weight": 0.5,
        },
        "american_professional": {
            "description": "Clear, precise American accent for STEM topics",
            "exaggeration": 0.5,  # Neutral
            "cfg_weight": 0.5,
        },
        "warm_teacher": {
            "description": "Warm, approachable voice for soft skills content",
            "exaggeration": 0.6,  # Slightly more expressive
            "cfg_weight": 0.5,
        },
    }

    def __init__(self, openai_api_key: str = None, runpod_api_key: str = None,
                 runpod_endpoint_id: str = None):
        self.output_dir = "data/outputs"
        self.temp_dir = "data/temp"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # API keys
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY', '')
        self.runpod_api_key = runpod_api_key or os.environ.get('RUNPOD_API_KEY', '')
        self.runpod_endpoint_id = runpod_endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID', '5ncsdkoclqv8yj')

        # RunPod endpoint URL
        self.runpod_url = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}"

        print(f"[VideoGenerator] RunPod configured: {'✓' if self.runpod_api_key else '✗'}")
        print(f"[VideoGenerator] RunPod Endpoint: {self.runpod_endpoint_id}")

    def generate_voice(self, script: dict, voice_style: str, job_id: str,
                      progress_callback: Optional[Callable] = None) -> str:
        """
        Generate voice narration using RunPod Chatterbox TTS.

        Chatterbox TTS features:
        - Zero-shot voice cloning
        - Emotion/exaggeration control
        - Outperforms ElevenLabs in evaluations
        """
        print(f"[VOICE] Generating narration for job {job_id}")
        print(f"[VOICE] Style: {voice_style}")

        # Get voice profile config
        profile = self.VOICE_PROFILES.get(voice_style, self.VOICE_PROFILES["british_documentary"])
        print(f"[VOICE] Profile: {profile['description']}")

        # Collect text from script
        full_text = self._extract_script_text(script)
        print(f"[VOICE] Script length: {len(full_text)} characters ({len(full_text.split())} words)")

        if progress_callback:
            progress_callback(10)

        output_file = os.path.join(self.output_dir, f"{job_id}_voice.mp3")

        # Use RunPod Chatterbox TTS
        if self.runpod_api_key:
            success = self._generate_with_runpod(full_text, profile, output_file, progress_callback)
            if success:
                return output_file
            print("[VOICE] RunPod failed, trying gTTS fallback...")

        # Fallback to gTTS
        success = self._generate_with_gtts(full_text, output_file, progress_callback)
        if success:
            return output_file

        # Last resort - create silent placeholder
        print("[VOICE] ✗ All TTS engines failed, creating placeholder")
        self._create_silent_audio(output_file, 60)
        return output_file

    def _extract_script_text(self, script: dict) -> str:
        """Extract all text content from script structure."""
        all_text = []

        # Title
        title = script.get('title', '')
        if title:
            all_text.append(f"Welcome to this comprehensive guide on {title}.")

        # Content if provided
        content = script.get('content', '')
        if content:
            all_text.append(content)

        # Modules
        modules = script.get('modules', [])
        for i, module in enumerate(modules, 1):
            module_title = module.get('module', module.get('title', ''))
            if module_title:
                all_text.append(f"Module {i}: {module_title}.")

            # Sections
            for section in module.get('sections', []):
                section_title = section.get('title', '')
                section_content = section.get('content', section.get('description', ''))

                if section_title:
                    all_text.append(section_title)
                if section_content:
                    all_text.append(section_content)

        # Add closing
        if title:
            all_text.append(f"This concludes our guide on {title}. "
                          "For hands-on learning experience, visit Virtual Campus at eon reality dot com.")

        return " ".join(all_text)

    def _chunk_text(self, text: str, max_chars: int = 1000) -> list:
        """
        Chunk text into segments for TTS processing.
        Preserves sentence boundaries where possible.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _generate_with_runpod(self, text: str, profile: dict,
                               output_file: str, progress_callback: Callable) -> bool:
        """Generate audio using RunPod Chatterbox TTS."""
        print("[VOICE] Using RunPod Chatterbox TTS")

        try:
            # Chunk text for processing
            chunks = self._chunk_text(text, max_chars=1000)
            audio_segments = []

            print(f"[VOICE] Processing {len(chunks)} chunks...")

            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback(10 + int(70 * (i / len(chunks))))

                print(f"[VOICE] Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")

                # Submit job to RunPod
                response = requests.post(
                    f"{self.runpod_url}/run",
                    headers={
                        "Authorization": f"Bearer {self.runpod_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "input": {
                            "text": chunk,
                            "exaggeration": profile.get('exaggeration', 0.5),
                            "cfg_weight": profile.get('cfg_weight', 0.5)
                        }
                    },
                    timeout=30
                )

                if response.status_code != 200:
                    print(f"[VOICE] RunPod submit error: {response.status_code} - {response.text}")
                    return False

                job_data = response.json()
                job_id = job_data.get('id')

                if not job_id:
                    print(f"[VOICE] No job ID returned")
                    return False

                # Poll for completion
                audio_data = self._poll_runpod_job(job_id)

                if audio_data:
                    # Save segment
                    segment_file = os.path.join(self.temp_dir, f"segment_{i:04d}.wav")

                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data)
                    with open(segment_file, 'wb') as f:
                        f.write(audio_bytes)

                    audio_segments.append(segment_file)
                else:
                    print(f"[VOICE] Failed to get audio for chunk {i+1}")
                    return False

            # Concatenate all segments
            if audio_segments:
                if progress_callback:
                    progress_callback(85)

                self._concatenate_audio(audio_segments, output_file)

                if progress_callback:
                    progress_callback(100)

                print(f"[VOICE] ✓ RunPod audio saved to {output_file}")
                return True

            return False

        except Exception as e:
            print(f"[VOICE] RunPod error: {e}")
            return False

    def _poll_runpod_job(self, job_id: str, timeout: int = 300) -> Optional[str]:
        """Poll RunPod job until completion."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.runpod_url}/status/{job_id}",
                    headers={"Authorization": f"Bearer {self.runpod_api_key}"},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')

                    if status == 'COMPLETED':
                        output = data.get('output', {})
                        # Chatterbox returns audio as base64
                        return output.get('audio_base64') or output.get('audio')
                    elif status == 'FAILED':
                        print(f"[VOICE] RunPod job failed: {data.get('error')}")
                        return None
                    elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                        time.sleep(2)
                    else:
                        print(f"[VOICE] Unknown status: {status}")
                        time.sleep(2)
                else:
                    print(f"[VOICE] Poll error: {response.status_code}")
                    time.sleep(2)

            except Exception as e:
                print(f"[VOICE] Poll exception: {e}")
                time.sleep(2)

        print("[VOICE] RunPod job timeout")
        return None

    def _generate_with_gtts(self, text: str, output_file: str,
                            progress_callback: Callable) -> bool:
        """Generate audio using Google TTS (free fallback)."""
        print("[VOICE] Using gTTS (fallback)")

        try:
            from gtts import gTTS

            # Limit text for gTTS
            if len(text) > 5000:
                text = text[:5000] + "... This concludes the preview."

            if progress_callback:
                progress_callback(30)

            tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)

            if progress_callback:
                progress_callback(70)

            tts.save(output_file)

            if progress_callback:
                progress_callback(100)

            print(f"[VOICE] ✓ gTTS audio saved to {output_file}")
            return True

        except Exception as e:
            print(f"[VOICE] gTTS error: {e}")
            return False

    def _concatenate_audio(self, segments: list, output_file: str):
        """Concatenate audio segments using FFmpeg."""
        if not segments:
            return

        if len(segments) == 1:
            # Just convert single file to MP3
            subprocess.run([
                'ffmpeg', '-y', '-i', segments[0],
                '-codec:a', 'libmp3lame', '-qscale:a', '2',
                output_file
            ], capture_output=True, timeout=300)
            return

        # Create concat file
        concat_file = os.path.join(self.temp_dir, "concat_audio.txt")
        with open(concat_file, 'w') as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        # Concatenate and convert to MP3
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-codec:a', 'libmp3lame', '-qscale:a', '2',
            output_file
        ], capture_output=True, timeout=600)

    def _create_silent_audio(self, output_file: str, duration_seconds: int):
        """Create a silent audio file as placeholder."""
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'anullsrc=r=24000:cl=mono',
            '-t', str(duration_seconds),
            '-c:a', 'libmp3lame', '-q:a', '5',
            output_file
        ], capture_output=True, timeout=60)

    def generate_visuals(self, script: dict, style: str, frequency: int,
                        job_id: str, progress_callback: Optional[Callable] = None) -> list:
        """
        Generate visual assets for the video.
        Uses DALL-E 3 if available, otherwise creates gradient images.
        """
        print(f"[VISUAL] Generating images for job {job_id}")
        print(f"[VISUAL] Style: {style}, Frequency: every {frequency}s")

        # Calculate images needed
        duration_hours = script.get('target_duration_hours', 1)
        total_seconds = duration_hours * 3600
        num_images = max(1, int(total_seconds // frequency))
        num_images = min(num_images, 30)  # Cap for reasonable generation time

        print(f"[VISUAL] Generating {num_images} images...")

        visuals = []
        images_dir = os.path.join(self.temp_dir, job_id)
        os.makedirs(images_dir, exist_ok=True)

        title = script.get('title', 'Educational Content')
        modules = script.get('modules', [])

        for i in range(num_images):
            if progress_callback:
                progress_callback(int((i + 1) / num_images * 100))

            image_path = os.path.join(images_dir, f"img_{i:03d}.png")

            # Determine prompt based on module
            if modules and i < len(modules):
                module = modules[i]
                prompt = f"Professional educational illustration for: {module.get('module', title)}"
            else:
                prompt = f"Professional educational illustration for: {title}"

            # Try DALL-E 3 first
            if self.openai_api_key:
                success = self._generate_dalle_image(prompt, image_path, style)
                if success:
                    visuals.append({
                        "path": image_path,
                        "timestamp": i * frequency,
                        "description": prompt
                    })
                    continue

            # Fallback to gradient image
            self._create_gradient_image(image_path, title, i)
            visuals.append({
                "path": image_path,
                "timestamp": i * frequency,
                "description": f"Gradient visual for {title}"
            })

            time.sleep(0.1)

        print(f"[VISUAL] ✓ Generated {len(visuals)} images")
        return visuals

    def _generate_dalle_image(self, prompt: str, output_path: str, style: str) -> bool:
        """Generate image using DALL-E 3."""
        try:
            style_prompts = {
                "professional": "clean professional stock photo style, minimalist, corporate colors",
                "ai_generated": "digital art style, vibrant colors, futuristic",
                "infographic": "infographic style, data visualization, diagrams",
                "mixed": "educational illustration style"
            }

            full_prompt = f"{prompt}, {style_prompts.get(style, style_prompts['professional'])}, 1920x1080, high resolution"

            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": full_prompt,
                    "n": 1,
                    "size": "1792x1024",
                    "quality": "standard"
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                image_url = data['data'][0]['url']

                img_response = requests.get(image_url, timeout=60)
                if img_response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(img_response.content)
                    return True

            return False

        except Exception as e:
            print(f"[VISUAL] DALL-E error: {e}")
            return False

    def _create_gradient_image(self, output_path: str, title: str, index: int):
        """Create gradient image with text as fallback."""
        if not PIL_AVAILABLE:
            with open(output_path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')
            return

        try:
            width, height = 1920, 1080
            img = Image.new('RGB', (width, height))

            # EON brand colors
            colors = [
                ((30, 60, 114), (42, 82, 152)),
                ((20, 80, 100), (40, 120, 140)),
                ((40, 60, 80), (60, 90, 120)),
                ((50, 40, 90), (80, 60, 130)),
                ((30, 70, 60), (50, 100, 80)),
            ]

            c1, c2 = colors[index % len(colors)]

            for y in range(height):
                r = int(c1[0] + (c2[0] - c1[0]) * y / height)
                g = int(c1[1] + (c2[1] - c1[1]) * y / height)
                b = int(c1[2] + (c2[2] - c1[2]) * y / height)
                for x in range(width):
                    img.putpixel((x, y), (r, g, b))

            draw = ImageDraw.Draw(img)

            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            except:
                font_large = ImageFont.load_default()
                font_small = font_large

            text = title[:50] if title else "Educational Content"
            bbox = draw.textbbox((0, 0), text, font=font_large)
            text_width = bbox[2] - bbox[0]
            draw.text(((width - text_width) / 2, height / 2 - 60), text, fill='white', font=font_large)

            section_text = f"Section {index + 1}"
            bbox = draw.textbbox((0, 0), section_text, font=font_small)
            text_width = bbox[2] - bbox[0]
            draw.text(((width - text_width) / 2, height / 2 + 30), section_text, fill='lightgray', font=font_small)

            try:
                brand_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                brand_font = font_small
            draw.text((50, height - 50), "EON DeepLearn™", fill='white', font=brand_font)

            img.save(output_path, 'PNG', optimize=True)

        except Exception as e:
            print(f"[VISUAL] Image creation error: {e}")
            with open(output_path, 'wb') as f:
                f.write(b'\x00' * 100)

    def assemble_video(self, voice_file: str, visuals: list, job_id: str,
                      progress_callback: Optional[Callable] = None) -> str:
        """
        Assemble final video from voice and visuals using FFmpeg.
        Per spec: -c:v libx264 -tune stillimage -c:a aac -b:a 192k
        """
        print(f"[VIDEO] Assembling video for job {job_id}")

        output_file = os.path.join(self.output_dir, f"{job_id}_final.mp4")

        if progress_callback:
            progress_callback(10)

        # Get audio duration
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', voice_file],
                capture_output=True, text=True, timeout=30
            )
            audio_duration = float(result.stdout.strip())
            print(f"[VIDEO] Audio duration: {audio_duration:.1f} seconds")
        except:
            audio_duration = 60
            print(f"[VIDEO] Using default duration: {audio_duration}s")

        if progress_callback:
            progress_callback(20)

        if not visuals:
            print("[VIDEO] ✗ No visuals to assemble")
            return None

        num_images = len(visuals)
        duration_per_image = audio_duration / num_images
        print(f"[VIDEO] {num_images} images, {duration_per_image:.1f}s each")

        if progress_callback:
            progress_callback(30)

        # Create concat file for slideshow
        concat_file = os.path.join(self.temp_dir, f"{job_id}_concat.txt")
        with open(concat_file, 'w') as f:
            for v in visuals:
                escaped_path = v['path'].replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
                f.write(f"duration {duration_per_image}\n")
            if visuals:
                escaped_path = visuals[-1]['path'].replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        if progress_callback:
            progress_callback(40)

        # FFmpeg command per spec
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-i', voice_file,
            '-c:v', 'libx264',
            '-tune', 'stillimage',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-movflags', '+faststart',
            output_file
        ]

        print("[VIDEO] Running FFmpeg...")
        if progress_callback:
            progress_callback(50)

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )

            if result.returncode != 0:
                print(f"[VIDEO] FFmpeg error: {result.stderr[:500]}")
                return self._simple_video_assembly(voice_file, visuals, job_id, output_file, progress_callback)

            if progress_callback:
                progress_callback(100)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
                size_mb = os.path.getsize(output_file) / 1024 / 1024
                print(f"[VIDEO] ✓ Video saved: {output_file} ({size_mb:.1f} MB)")
                return output_file
            else:
                print("[VIDEO] ✗ Output file too small")
                return self._simple_video_assembly(voice_file, visuals, job_id, output_file, progress_callback)

        except subprocess.TimeoutExpired:
            print("[VIDEO] ✗ FFmpeg timeout")
            return None
        except Exception as e:
            print(f"[VIDEO] ✗ FFmpeg error: {e}")
            return self._simple_video_assembly(voice_file, visuals, job_id, output_file, progress_callback)

    def _simple_video_assembly(self, voice_file: str, visuals: list, job_id: str,
                               output_file: str, progress_callback: Callable) -> str:
        """Simple fallback assembly with single image loop."""
        print("[VIDEO] Trying simple assembly...")

        try:
            if visuals and os.path.exists(visuals[0]['path']):
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1',
                    '-i', visuals[0]['path'],
                    '-i', voice_file,
                    '-c:v', 'libx264',
                    '-tune', 'stillimage',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-pix_fmt', 'yuv420p',
                    '-shortest',
                    output_file
                ]

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0 and os.path.exists(output_file):
                    size_mb = os.path.getsize(output_file) / 1024 / 1024
                    if progress_callback:
                        progress_callback(100)
                    print(f"[VIDEO] ✓ Simple video saved: {output_file} ({size_mb:.1f} MB)")
                    return output_file

            print("[VIDEO] ✗ Simple assembly failed")
            return None

        except Exception as e:
            print(f"[VIDEO] ✗ Simple assembly error: {e}")
            return None


# Export the class
VideoGenerator = VideoGeneratorReal
