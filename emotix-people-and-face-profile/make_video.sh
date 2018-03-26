ffmpeg -start_number 0 -framerate 25 -i ./output/%00d.jpg -s:v 1920:1080 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./out_terrace.mp4
