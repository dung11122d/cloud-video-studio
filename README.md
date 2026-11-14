# 🎬 Cloud Video Studio (`dung11122d/cloud-video-studio`)

Hệ thống chỉnh sửa, render video và tối ưu SEO YouTube tự động **100% trên đám mây (GitHub Actions 4-vCPU / 16GB RAM & Google Colab GPU)** — **Không tốn 1% CPU/RAM nào trên máy tính cá nhân**.

---

## 🚀 Cách 1: Chỉnh sửa Video trực tiếp từ Antigravity (hoặc 1 nút bấm trên Web)

### A. Chạy ngay từ khung chat Antigravity (Khuyên dùng)
Chỉ cần nhắn cho Antigravity hoặc chạy lệnh `gh workflow run`:
```powershell
D:\Antigravity\tools\gh\bin\gh.exe workflow run cloud-video-editor.yml `
  --repo dung11122d/cloud-video-studio `
  -f video_source="https://www.youtube.com/watch?v=VIDEO_ID" `
  -f trim_start="00:00:10" `
  -f trim_end="00:01:10" `
  -f remove_silence="true" `
  -f aspect_mode="9:16-blur" `
  -f seo_keyword="hướng dẫn làm video ai"
```

### B. Chạy trên giao diện Web GitHub
1. Mở tab **Actions** của repo: `https://github.com/dung11122d/cloud-video-studio/actions`
2. Chọn workflow **🎬 Cloud Video Editor & YouTube Auto-Upload** $\rightarrow$ bấm **Run workflow**.
3. Điền link video, chọn cắt khoảng lặng, chọn khung hình `9:16-blur` (Shorts/TikTok) hoặc `original`, nhập từ khóa SEO $\rightarrow$ bấm **Run workflow**.
4. Sau 1–2 phút, tải file `final_video.mp4` và `meta.json` ngay tại mục **Artifacts** (hoặc bật tự động đăng thẳng lên kênh YouTube).
