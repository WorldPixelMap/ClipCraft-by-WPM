"""Unit and integration tests for Local ClipCraft by WPM."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.caption_engine import (
    devanagari_to_hinglish,
    get_clip_words,
    is_offline_model_cached,
    parse_custom_script_or_srt,
    write_highlighted_ass_and_srt,
)
from app.models import (
    AspectRatio,
    CaptionEngineMode,
    CaptionLanguage,
    CornerPosition,
    HighlightColor,
    MomentCandidate,
    QualityPreset,
    TranscriptSegment,
    WordTimestamp,
)
from app.moment_detector import detect_best_moments
from app.face_tracker import (
    _build_active_intervals,
    build_ffmpeg_lerp_expr,
    is_face_model_cached,
    track_corner_streamer_face,
    track_main_speaker_9_16,
)
from app.social_suggester import suggest_social_metadata
from app.video_renderer import _cam_source_crop, _pip_dimensions, probe_video, render_clip
from app.youtube_auth import cookies_to_netscape_text, find_chromium_browser
from app.youtube_service import (
    _base_ydl_opts,
    find_ffmpeg_dir,
    find_node_exe,
    format_ytdlp_error,
    parse_json3_subtitles,
    parse_vtt_subtitles,
    validate_video_url,
    validate_youtube_url,
)


def _ffmpeg_exe() -> str:
    ffdir = find_ffmpeg_dir()
    return str(Path(ffdir) / "ffmpeg.exe") if ffdir else "ffmpeg"


class TestUrlAndSubtitles(unittest.TestCase):
    def test_validate_youtube_and_multiplatform_urls(self) -> None:
        self.assertEqual(
            validate_youtube_url("https://youtu.be/jNQXAC9IVRw?t=5"),
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        self.assertEqual(
            validate_youtube_url("https://www.youtube.com/shorts/abcdefghijk"),
            "https://www.youtube.com/watch?v=abcdefghijk",
        )
        self.assertEqual(
            validate_youtube_url("youtube.com/watch?v=dQw4w9WgXcQ&list=PL123"),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        twitch_url, twitch_plat = validate_video_url("https://www.twitch.tv/videos/123456789")
        self.assertEqual((twitch_url, twitch_plat), ("https://www.twitch.tv/videos/123456789", "Twitch"))

        kick_url, kick_plat = validate_video_url("https://kick.com/streamer/clips/clip_123")
        self.assertEqual(kick_plat, "Kick")
        self.assertTrue(kick_url.startswith("https://kick.com/"))

        with self.assertRaises(ValueError):
            validate_video_url("http://127.0.0.1:8000/secret")
        with self.assertRaises(ValueError):
            validate_video_url("http://localhost/admin")

        friendly = format_ytdlp_error(RuntimeError("Sign in to confirm you’re not a bot"), "YouTube")
        self.assertIn("Sign in to YouTube", friendly)

    def test_cdp_cookies_to_netscape_filters_and_detects_auth(self) -> None:
        raw = [
            {"domain": ".youtube.com", "name": "LOGIN_INFO", "value": "tok123", "path": "/", "secure": True, "expires": 1999999999},
            {"domain": ".google.com", "name": "SAPISID", "value": "sap456", "path": "/", "secure": True, "expires": 0},
            {"domain": ".evil.example.com", "name": "TRACK", "value": "secret", "path": "/", "secure": False, "expires": 1999999999},
        ]
        txt, has_auth = cookies_to_netscape_text(raw)
        self.assertTrue(has_auth)
        self.assertIn("# Netscape HTTP Cookie File", txt)
        self.assertIn(".youtube.com\tTRUE\t/\tTRUE\t1999999999\tLOGIN_INFO\ttok123", txt)
        self.assertNotIn("evil.example.com", txt)
        self.assertIsNotNone(find_chromium_browser())
        self.assertIsNotNone(find_node_exe())
        self.assertIn("node", _base_ydl_opts().get("js_runtimes", {}))

    def test_parse_json3_with_word_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.json3"
            payload = {
                "events": [
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 2000,
                        "segs": [
                            {"utf8": "Here"},
                            {"utf8": " is", "tOffsetMs": 300},
                            {"utf8": " the secret!", "tOffsetMs": 600},
                        ],
                    }
                ]
            }
            p.write_text(json.dumps(payload), encoding="utf-8")
            segs = parse_json3_subtitles(p)
            self.assertEqual(len(segs), 1)
            self.assertTrue(segs[0].has_word_timing)
            self.assertEqual([w.word for w in segs[0].words], ["Here", "is", "the", "secret!"])
            for i in range(len(segs[0].words) - 1):
                self.assertLess(segs[0].words[i].start, segs[0].words[i + 1].start)

    def test_json3_deduplicates_dual_pen_and_zero_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dup.json3"
            payload = {
                "events": [
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 1500,
                        "segs": [
                            {"utf8": "\u200b \u200bWait, ", "pPenId": 3},
                            {"utf8": "is ", "pPenId": 3, "tOffsetMs": 250},
                            {"utf8": "there", "pPenId": 3, "tOffsetMs": 500},
                        ],
                    },
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 1500,
                        "segs": [
                            {"utf8": "\u200b \u200bWait, ", "pPenId": 4},
                            {"utf8": "is ", "pPenId": 4, "tOffsetMs": 250},
                            {"utf8": "there", "pPenId": 4, "tOffsetMs": 500},
                        ],
                    },
                ]
            }
            p.write_text(json.dumps(payload), encoding="utf-8")
            segs = parse_json3_subtitles(p)
            self.assertEqual(len(segs), 1)
            self.assertEqual(segs[0].text, "Wait, is there")
            self.assertNotIn("\u200b", segs[0].text)

    def test_parse_vtt_rolling_and_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.vtt"
            p.write_text(
                "WEBVTT\n\n"
                "00:00:01.000 --> 00:00:03.000\n"
                "Hello<00:00:01.400><c> world</c><00:00:02.000><c> today</c>\n\n"
                "00:00:03.000 --> 00:00:03.010\n"
                "Hello world today\n\n"
                "00:00:03.010 --> 00:00:05.000\n"
                "Hello world today\n"
                "how<00:00:03.500><c> are</c><00:00:04.000><c> you</c>\n",
                encoding="utf-8",
            )
            segs = parse_vtt_subtitles(p)
            self.assertEqual(len(segs), 2)
            self.assertEqual([w.word for w in segs[0].words], ["Hello", "world", "today"])
            self.assertEqual([w.word for w in segs[1].words], ["how", "are", "you"])


class TestHindiHinglishAndSocial(unittest.TestCase):
    def test_devanagari_to_hinglish_and_social_suggestions(self) -> None:
        out = devanagari_to_hinglish("नमस्ते दोस्तों! आज यह वीडियो बहुत अच्छा है।")
        self.assertIn("aaj", out.lower())
        self.assertIn("yeh", out.lower())
        self.assertIn("accha", out.lower())
        self.assertIn("hai", out.lower())

        moment = MomentCandidate(
            rank=1,
            start=10.0,
            end=45.0,
            duration=35.0,
            score=92.5,
            reasons=["High Vocal Energy", "Strong Opening Hook"],
        )
        social = suggest_social_metadata(
            clip_text="Why do 90% of streamers miss this clutch rotation? Never push alone without vision!",
            video_title="Pro Valorant Clutch Guide",
            channel="WPM Gaming",
            platform="Twitch",
            moment=moment,
            video_tags=["Valorant", "Gaming", "Clutch"],
            caption_language=CaptionLanguage.HINGLISH,
        )
        self.assertTrue(len(social.title) <= 72)
        self.assertIn("?", social.title)
        self.assertIn("#Twitch", social.hashtags)
        self.assertIn("#Hinglish", social.hashtags)
        self.assertIn("Title:", social.copy_all_youtube)
        self.assertIn("Title:", social.copy_all_facebook)


class TestSyntheticClippingPipeline(unittest.TestCase):
    def test_moment_detection_corner_pip_hindi_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "synth.mp4"
            cmd = [
                _ffmpeg_exe(),
                "-v", "error",
                "-y",
                "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=28",
                "-f", "lavfi", "-i",
                "aevalsrc='if(between(t,8,19),0.7*sin(2*PI*440*t),0.04*sin(2*PI*220*t))':s=44100:d=28",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(src),
            ]
            subprocess.run(cmd, check=True)

            segments = [
                TranscriptSegment(text="Welcome to the channel intro.", start=1.0, end=5.5),
                TranscriptSegment(
                    text="Here's why this secret actually changes everything! Never make this mistake.",
                    start=8.2,
                    end=18.2,
                ),
                TranscriptSegment(text="Thanks for watching the outro.", start=21.0, end=26.5),
            ]
            moments, audio = detect_best_moments(
                video_path=src,
                duration=28.0,
                num_clips=2,
                target_duration=10,
                heatmap=[
                    {"start_time": 0.0, "end_time": 8.0, "value": 0.1},
                    {"start_time": 8.0, "end_time": 19.0, "value": 0.95},
                    {"start_time": 19.0, "end_time": 28.0, "value": 0.2},
                ],
                segments=segments,
            )
            self.assertEqual(len(moments), 2)
            self.assertGreaterEqual(moments[0].end, 14.0)
            self.assertLessEqual(moments[0].start, 11.0)

            words, cap_src, ai_used = get_clip_words(
                video_path=src,
                segments=segments,
                clip_start=moments[0].start,
                clip_end=moments[0].end,
                audio=audio,
                mode=CaptionEngineMode.NO_AI,
                work_dir=work,
                rank=1,
                caption_language=CaptionLanguage.ENGLISH,
            )
            self.assertFalse(ai_used)
            self.assertEqual(cap_src, "youtube_audio_paced")

            ass_path = work / "clip_1.ass"
            srt_path = work / "clip_1.srt"
            write_highlighted_ass_and_srt(
                words=words,
                ass_path=ass_path,
                srt_path=srt_path,
                video_w=720,
                video_h=1280,
                clip_duration=moments[0].duration,
                aspect_ratio=AspectRatio.VERTICAL_CROP,
                highlight_color=HighlightColor.YELLOW,
                caption_language=CaptionLanguage.ENGLISH,
                watermark_text="@WPM",
            )
            ass_content = ass_path.read_text(encoding="utf-8")
            self.assertIn("Local ClipCraft by WPM", ass_content)
            self.assertIn("@WPM", ass_content)

            # Test Corner PiP + 9:16 crop + audio boost
            out_mp4 = work / "clip_1.mp4"
            out_jpg = work / "clip_1.jpg"
            src_info = probe_video(src)
            render_clip(
                source_video=src,
                src_info=src_info,
                output_mp4=out_mp4,
                output_thumb=out_jpg,
                start=moments[0].start,
                end=moments[0].end,
                aspect_ratio=AspectRatio.VERTICAL_CROP,
                quality=QualityPreset.FAST,
                ass_path=ass_path,
                work_dir=work,
                streamer_cam_source=CornerPosition.BOTTOM_LEFT,
                streamer_cam_target=CornerPosition.TOP_RIGHT,
                audio_boost=True,
            )
            self.assertTrue(out_mp4.exists())
            rendered_info = probe_video(out_mp4)
            self.assertEqual((rendered_info.width, rendered_info.height), (720, 1280))

            # Also verify 9:16-fit blur + Hindi Devanagari ASS burn-in
            hi_ass = work / "clip_hi.ass"
            hi_words = [
                WordTimestamp(word="नमस्ते", start=0.2, end=1.0),
                WordTimestamp(word="दोस्तों", start=1.0, end=2.0),
            ]
            write_highlighted_ass_and_srt(
                words=hi_words,
                ass_path=hi_ass,
                srt_path=None,
                video_w=720,
                video_h=1280,
                clip_duration=5.0,
                aspect_ratio=AspectRatio.VERTICAL_FIT,
                highlight_color=HighlightColor.CYAN,
                caption_language=CaptionLanguage.HINDI,
            )
            self.assertIn("Nirmala UI", hi_ass.read_text(encoding="utf-8"))
            out_fit = work / "clip_fit.mp4"
            render_clip(
                source_video=src,
                src_info=src_info,
                output_mp4=out_fit,
                output_thumb=work / "clip_fit.jpg",
                start=0.0,
                end=5.0,
                aspect_ratio=AspectRatio.VERTICAL_FIT,
                quality=QualityPreset.FAST,
                ass_path=hi_ass,
                work_dir=work,
                streamer_cam_source=CornerPosition.TOP_RIGHT,
                streamer_cam_target=CornerPosition.BOTTOM_LEFT,
                audio_boost=False,
            )
            self.assertEqual((probe_video(out_fit).width, probe_video(out_fit).height), (720, 1280))

    def test_offline_model_cached(self) -> None:
        self.assertTrue(is_offline_model_cached())

    def test_pip_cam_is_tiny_9_16_ratio(self) -> None:
        pw, ph = _pip_dimensions(720, 1280)
        self.assertAlmostEqual(pw / ph, 9.0 / 16.0, places=2)
        self.assertLessEqual(pw, 170)

        # Adaptive face-fitted Cam Frame matches crop_ar without blurred bars
        apw, aph = _pip_dimensions(720, 1280, crop_ar=0.82)
        self.assertAlmostEqual(apw / aph, 0.82, delta=0.04)
        self.assertLessEqual(apw, 210)
        sqw, sqh = _pip_dimensions(720, 1280, crop_ar=1.0)
        self.assertAlmostEqual(sqw / sqh, 1.0, delta=0.04)

        crop = _cam_source_crop(1280, 720, CornerPosition.TOP_LEFT)
        parts = [int(x) for x in crop.replace("crop=", "").split(":")]
        cw, ch = parts[0], parts[1]
        self.assertAlmostEqual(cw / ch, 9.0 / 16.0, places=2)

    def test_face_tracker_corner_pip_and_main_9_16_expr(self) -> None:
        self.assertTrue(is_face_model_cached())
        # Simulate moving face inside top-left corner + speaker on right side of main frame
        frame_faces = []
        for idx in range(8):
            t = idx * 0.5
            # Corner streamer face around x=180..220, y=50..120 (shifted from naive edge!)
            corner_face = (170.0 + idx * 5.0, 55.0 + idx * 2.0, 62.0, 72.0, 0.92)
            # Main scene speaker on far right side (x=960)
            main_face = (950.0, 180.0, 110.0, 130.0, 0.95)
            frame_faces.append((t, [corner_face, main_face]))

        # When requested explicitly in TOP_LEFT, tracks TOP_LEFT cleanly
        corner_track = track_corner_streamer_face(frame_faces, 1280, 720, CornerPosition.TOP_LEFT)
        self.assertIsNotNone(corner_track)
        self.assertEqual(corner_track.detected_corner, CornerPosition.TOP_LEFT)
        crop_ar = corner_track.cw / corner_track.ch
        self.assertGreaterEqual(crop_ar, 0.56)
        self.assertLessEqual(crop_ar, 0.90)
        pw, ph = _pip_dimensions(720, 1280, crop_ar=crop_ar)
        self.assertAlmostEqual(pw / ph, crop_ar, delta=0.04)
        # Crop center X must track the streamer face near x ~ 200 rather than blind x=93 edge
        mid_kf = corner_track.keyframes[len(corner_track.keyframes) // 2]
        self.assertGreater(mid_kf.x + corner_track.cw / 2.0, 175.0)
        self.assertLess(mid_kf.x + corner_track.cw / 2.0, 245.0)

        main_track = track_main_speaker_9_16(frame_faces, 1280, 720, exclude_corner=CornerPosition.TOP_LEFT)
        self.assertIsNotNone(main_track)
        self.assertGreater(main_track.keyframes[0].x, 700.0)

        expr_x = build_ffmpeg_lerp_expr(corner_track.keyframes, axis="x")
        self.assertIn("if(lt(t,", expr_x)

        # Jump-cut snapping must step instead of gliding smoothly across huge jumps
        snap_expr = build_ffmpeg_lerp_expr(
            [
                corner_track.keyframes[0],
                corner_track.keyframes[-1].__class__(t=2.0, x=980.0, y=500.0),
            ],
            axis="x",
            jump_thresh=80.0,
        )
        self.assertIn("if(lt(t,1.00)", snap_expr)
        self.assertNotIn("*(t-", snap_expr)

        # Cutscene gap > 1.2s splits active intervals so empty cutscene PiP stays hidden
        intervals = _build_active_intervals([0.0, 0.5, 1.0, 5.0, 5.5, 6.0], clip_dur=8.0)
        self.assertEqual(len(intervals), 2)

        # Verify FFmpeg executes dynamic multi-keyframe lerp crop expression cleanly
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out = work / "tracked.mp4"
            cmd = [
                _ffmpeg_exe(),
                "-v", "error",
                "-y",
                "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=4",
                "-vf", f"crop={corner_track.cw}:{corner_track.ch}:'{expr_x}':20,scale={pw}:{ph}",
                "-c:v", "libx264", "-preset", "ultrafast",
                str(out),
            ]
            subprocess.run(cmd, check=True)
            info = probe_video(out)
            self.assertEqual((info.width, info.height), (pw, ph))

    def test_face_tracker_anywhere_positions_and_anti_npc(self) -> None:
        # 1. Bottom-Center streamer + transient high-conf NPC walking across middle
        frame_faces = []
        for idx in range(10):
            t = idx * 0.4
            bot_center_streamer = (605.0, 560.0, 68.0, 82.0, 0.89)
            faces = [bot_center_streamer]
            if idx in (2, 3, 4):
                # Roaming in-game NPC near center crosshair
                faces.append((540.0 + idx * 38.0, 260.0 + idx * 14.0, 72.0, 86.0, 0.94))
            frame_faces.append((t, faces))

        tr = track_corner_streamer_face(frame_faces, 1280, 720, CornerPosition.AUTO)
        self.assertIsNotNone(tr)
        self.assertEqual(tr.detected_corner, CornerPosition.BOTTOM_CENTER)
        mid = tr.keyframes[len(tr.keyframes) // 2]
        self.assertAlmostEqual(mid.x + tr.cw / 2.0, 639.0, delta=25.0)
        self.assertAlmostEqual(mid.y + tr.ch / 2.0, 601.0, delta=35.0)

        # 2. Mid-Left streamer position
        mid_left_frames = [(idx * 0.5, [(72.0, 315.0 + idx * 2.0, 66.0, 80.0, 0.91)]) for idx in range(6)]
        tr_ml = track_corner_streamer_face(mid_left_frames, 1280, 720, CornerPosition.AUTO)
        self.assertIsNotNone(tr_ml)
        self.assertEqual(tr_ml.detected_corner, CornerPosition.MID_LEFT)

        # 3. Top-Center streamer position
        top_ctr_frames = [(idx * 0.5, [(595.0, 48.0, 74.0, 88.0, 0.92)]) for idx in range(6)]
        tr_tc = track_corner_streamer_face(top_ctr_frames, 1280, 720, CornerPosition.AUTO)
        self.assertIsNotNone(tr_tc)
        self.assertEqual(tr_tc.detected_corner, CornerPosition.TOP_CENTER)

        # 4. Dead-Center streamer position (IRL / full reaction)
        ctr_frames = [(idx * 0.5, [(575.0, 275.0, 125.0, 155.0, 0.94)]) for idx in range(6)]
        tr_c = track_corner_streamer_face(ctr_frames, 1280, 720, CornerPosition.AUTO)
        self.assertIsNotNone(tr_c)
        self.assertEqual(tr_c.detected_corner, CornerPosition.CENTER)
        c_mid = tr_c.keyframes[len(tr_c.keyframes) // 2]
        self.assertAlmostEqual(c_mid.x + tr_c.cw / 2.0, 637.5, delta=25.0)
        self.assertAlmostEqual(c_mid.y + tr_c.ch * 0.45, 352.5, delta=25.0)

    def test_custom_script_and_srt_zero_ai_pacing(self) -> None:
        srt_text = (
            "1\n00:00:00,200 --> 00:00:02,000\nWelcome to World Pixel Map\n\n"
            "2\n00:00:02,200 --> 00:00:04,500\nWatch this quick edit!\n"
        )
        words = parse_custom_script_or_srt(srt_text, duration=6.0)
        self.assertEqual(len(words), 9)
        self.assertGreaterEqual(words[0].start, 0.2)
        self.assertLessEqual(words[-1].end, 4.55)

        plain_words = parse_custom_script_or_srt("नमस्ते दोस्तों आज हम नया रील देखेंगे", duration=5.0, caption_language=CaptionLanguage.HINGLISH)
        self.assertEqual(len(plain_words), 7)
        self.assertEqual(plain_words[1].word.lower(), "doston")


if __name__ == "__main__":
    unittest.main()
