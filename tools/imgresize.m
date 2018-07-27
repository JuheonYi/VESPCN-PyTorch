function imgresize()
folders = dir('./HR');
for i = 3:numel(folders)
    foldername = folders(i).name;
    folderpath = ['./HR/', foldername];
    files = dir(folderpath);
    newfoldpath = ['./LR_bicubic/X3/', foldername];
    mkdir(newfoldpath);
    for j = 3:numel(files)
        fname = files(j).name;
        disp(fname)
        fullname = fullfile(folderpath, fname);
        img = imread(fullname);
        lrimg = imresize(img, 1/3);
        newfpath = fullfile(newfoldpath, fname);
        imwrite(lrimg, newfpath);
    end
end
end