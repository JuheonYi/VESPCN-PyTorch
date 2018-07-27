function videocapture(videoname, time)
videon = ['video', videoname, '.avi'];
v = VideoReader(videon);

v.CurrentTime = time;
x = v.CurrentTime;
subfolder = ['Video', videoname];
mkdir('./HR', subfolder);
for ii = 1:30
   img = readFrame(v);
   filename = [sprintf('%05d',int32(ii+x * 30)) '.png'];
   fullname = fullfile('./HR',subfolder, filename);
   imwrite(img,fullname)
end
end
