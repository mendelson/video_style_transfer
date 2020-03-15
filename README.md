# video_style_transfer
A style transfer for videos based on Gatys' paper using PyTorch.

I used PyTorch and Google Colab (a lot of time due to timeouts), and I encourage you to apply different style images and videos.

If you find this repo useful, please ping me so we can have a cup of coffee if you ever come to my city.

## Using this code
To run the code, you must run the indicated cells in the __video_style.ipyng__ file.

The videos to be stylized must be inside the *input videos* folder. The style images must be in the *styles* folder.

You may change the file names as desired in the third code cell of the code.

Note that the *results* folder contains some sample results and it is not used from within the code.

## Folder structure
.
├── frames 										# holds the frames for the current video
│     ├── input frames 							# holds the input frames from the original video
│     └── style frames 							# holds the stylized frames, which will be used to make the stylized output video
├── input videos 								# holds some sample videos for input
├── results										# holds some sample results
│	 └── {input video name}						# there is one folder for each sample input video
│	 	  ├── original frames 					# holds the input frames from the original video
│	 	  ├── {name of style file} style 		# holds the stylized results
│	 	  │	   ├── style frames 				# holds the stylized frames, which were used to make the stylized output video
│	 	  │	   └── stylized_video.mp4 			# the stylized output video
│	 	  └── properties.pkl 					# pickle file that contains the input video's properties
└── styles 										# holds some sample style images