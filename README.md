# udacity-4-Dynamic-Risk-Assessment

Some additional comments.
I have used my local environment instead of your workspace. As in the other moduls of the course
I have made the solution available on github. You can access it here:
https://github.com/kangaroo98/udacity-4-Dynamic-Risk-Assessment.git


Step1:
I have changed the directory structure in the folder 'practicemodel' to test and demonstrate the merge capabilities.
Please also see the screenshot.

Step2:
I have introduce a new file with shared capabilities (shared.py).
Each trained model is versioned by a number, which is reflected in the model file name. Additional tests for the same version
model are stored with the same version number. The 'mode' in the underlying pydantic model shows, if it is a 'train' or 'test' score.
Currently the model with last/highest version number is considered to be deployed.

Step3:
When measuring the exe time e.g.logging should be of course disabled. Just to let you know that I know ;-) 

Step4:
Implemented different post versions for the prediction. Take the first one, which is my favourite.

Step5:
When initially switching to the 'sourcedata' the model score was very bad. So I decided to append the new data to the already existing data 
('practicedata'), which improved the score significantly. Therefore you will see all for files in the 'ingestedfiles.txt' (see the screenshot).
