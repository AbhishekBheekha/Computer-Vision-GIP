%% Setup the video reader and player
videoReader = vision.VideoFileReader('angdistest.mp4','VideoOutputDataType','inherit');
videoPlayer = vision.DeployableVideoPlayer('Size','Custom');

%% Setup track storage structure
tracks = struct('id', {},'centroid', {},'kalmanFilter');
nextId = 1;

%% Tracking 
frameNo = 0;
while ~isDone(videoReader)

    % Aquire single frame
    videoFrame = step(videoReader);
    
    % Apply detection function
    [~,centroid] = detection(videoFrame,1000);
    
    % Determine number of points
    points = size(centroid,1);

    % Insert marker at detected object center for one object
    % Calculate and store location of detected centre
    if points== 1 && ~isempty(centroid)  
       videoFrame = insertShape(videoFrame,'circle', [centroid 15], 'LineWidth',8);
       loc((2*frameNo+1),:) = centroid(1,:);
    else
       loc((2*frameNo+1),:) = [NaN NaN];
    end
    
    % Insert markers at detected object centres for two objects
    % Calculate and store angle and midpoint for two degree of freedom robot
    if (points==1) && ~isempty(centroid)  
       midpoint = [(0.5*sum(centroid(:,1))),(0.5*sum(centroid(:,2)))];
       videoFrame = insertShape(videoFrame,'circle', [centroid(1,:) 15], 'LineWidth',8);
       videoFrame = insertShape(videoFrame,'circle', [centroid(2,:) 15], 'LineWidth',8);
       theta = (90-(atan((centroid(1,2)-centroid(2,2))/(centroid(2,1)-centroid(1,1)))*(180/pi)));
       mid(frameNo+1,:) = midpoint;
    else
       mid(frameNo+1,:) = [NaN NaN];
       angle = theta();
    end
    
    % Acquire detection property
    tracks = length(tracks);
  
    cost = zeros(tracks, points);
    for currentId = 1:tracks
        % Predict next location for current tracks
        tracks(currentId).centroid = predict(tracks(currentId).kalmanFilter);
        % Calculate cost for assigning predicted detection to tracks
        cost(currentId,:) = distance(tracks(currentId).kalmanFilter, centroid);
    end
    
    %% Apply Munkres variation of the Hungarian algorithm 
    [assignments] = assignDetectionsToTracks(cost, 100);
    
    %% Update assigned tracks
    for currentId = 1:size(assignments, 1)
        trackId = assignments(currentId, 1);
        detectionCurrentId = assignments(currentId, 2);
        correct(tracks(trackId).kalmanFilter, centroid(detectionCurrentId, :));
    end
    
    %% Delete unassigned and lost tracks
    toBeDeleted = false(tracks,1);
    tracks(toBeDeleted) = [];
    
    % Display Frame
    step(videoPlayer,videoFrame);

    frameNo = frameNo + 1;

end

% Release video player and video reader     
release(videoPlayer);
release(videoReader);

%%% Dectection function %%%
function [regionarea,centroid] = detection(vidFrame,minArea)

%% To preserve the state of the frame for the next function call
persistent hBlob  

%% Blob Analysis
if isempty(hBlob)    
blobAnalysis = vision.BlobAnalysis('AreaOutputPort', true, ...
  'MinimumBlobArea', minArea, 'ExcludeBorderBlobs', true);
end

%% Image Processing
hsvImage = rgb2hsv(vidFrame);
sImage = hsvImage(:,:,2);
binaryImage = sImage>=0.8;

%% Morphological operations
binaryImage = imopen(binaryImage,strel('disk', 10));

% Return properties of regions with more than minArea pixels
[regionarea,centroid] = step(blobAnalysis, binaryImage);

end
