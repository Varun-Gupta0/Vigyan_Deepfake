def fusion_engine(video_score, text_score, face_count=None):
    video_score = float(video_score) if video_score is not None else 0.0
    text_score = float(text_score) if text_score is not None else 0.0

    has_video = video_score > 0 or (video_score == 0 and text_score == 0)
    has_text = text_score > 0

    if has_video and has_text:
        fused = 0.7 * video_score + 0.3 * text_score
    elif has_video:
        fused = video_score
    elif has_text:
        fused = text_score
    else:
        fused = 0.5

    if face_count is not None and face_count == 0 and has_video:
        fused = min(1.0, fused + 0.05)

    return max(0.0, min(1.0, fused))
