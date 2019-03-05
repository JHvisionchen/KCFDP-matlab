function run_tracker()

    addpath(genpath('./piotr_toolbox/toolbox/'));
    close all
    bSaveImage = 0;
    res_path = './result';
    if bSaveImage && ~exist(res_path,'dir')
       mkdir(res_path);
    end
    
    % choose the path to the videos (you'll be able to choose one with the GUI)
    base_path = '/media/cjh/datasets/tracking/OTB100';
    %ask the user for the video
    [video_path,video_name] = choose_video(base_path);
    text_files = dir([video_path 'groundtruth_rect.txt']);
    assert(~isempty(text_files), 'No initial position and ground truth (groundtruth_rect.txt) to load.')
    
    f = fopen([video_path text_files(1).name]);
    ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
    ground_truth = cat(2, ground_truth{:});
    fclose(f);

    startf = 1;
    endf = size(ground_truth,1);
    % config sequence:
    seq=struct('name',video_name,'path',video_path,'startFrame',startf,'endFrame',endf,'nz',4,'ext','jpg','init_rect', [0,0,0,0]);
    seq.len = seq.endFrame - seq.startFrame + 1;
    seq.s_frames = cell(seq.len,1);
    nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
    for i=1:seq.len
        image_no = seq.startFrame + (i-1);
        id = sprintf(nz,image_no);
        seq.s_frames{i} = strcat(seq.path,'img/',id,'.',seq.ext); % add 'img/' in every image path
    end
    rect_anno = dlmread([seq.path '/groundtruth_rect.txt']);
    seq.init_rect = rect_anno(seq.startFrame,:);
    s_frames = seq.s_frames;
    % parameters according to the paper
    padding = 1.5;  %extra area surrounding the target
    lambda = 1e-4;  %regularization
    % output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
    output_sigma_factor = 0.06;  %spatial bandwidth (proportional to target)
    % interp_factor = 0.02;
    interp_factor = 0.01;
    sigma = 0.5;
    hog_orientations = 9;
    cell_size = 4;

    target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
    pos = floor([seq.init_rect(1,2), seq.init_rect(1,1)]) + floor(target_sz/2);
    % general parameters
    params.visualization = 1;
    params.init_pos = pos;
    params.wsize = floor(target_sz);
    params.video_path = seq.path;
    params.s_frames = s_frames;
    params.bSaveImage = bSaveImage;
    params.res_path = res_path;
    % load pre-trained edge detection model and set opts
    model = load('./models/forest/modelBsds'); 
    model = model.model;
    model.opts.multiscale = 0;
    model.opts.sharpen = 0;
    model.opts.nThreads = 4;
    % set up parameters for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .65;      % step size of sliding window search
    opts.beta = .75;       % nms threshold for object proposals
    %opts.maxBoxes = 1e4;  % max number of boxes to detect
    opts.maxBoxes = 1e3;   % don't need that many proposals (default is 1e4)
    opts.minScore = 0.0005;
    opts.kappa = 1.4;     % 1.5 as default, can be changed for larger overlapping
    opts.minBoxArea = 200;
    opts.edgeMinMag = 0.1;
    % set up parameters for using edgeBoxes in scale detection
    scale_params.scale_detect_window_factor = 1.4;
    scale_params.proposal_num_limit = 200;
    scale_params.pos_shift_damping = 0.7;
    scale_params.rescale_damping = 0.7;

    [~, fps] = tracker(params, padding, sigma, lambda, output_sigma_factor, interp_factor,cell_size, hog_orientations, model, opts, scale_params);
    disp(['fps: ' num2str(fps)])

end
