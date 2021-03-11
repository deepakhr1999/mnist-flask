# MNIST Flask App
![](https://github.com/deepakhr1999/mnist-flask/blob/master/media/screenshot.png)  

A Flask web app for handwritten digits using a logistic regression model  
Used as an assignment follow up with Chapter-1: Basics, ML Workshop, AI-Club, IIT Dharwad

### Step 1: Download code
You can download this code using this link https://github.com/deepakhr1999/mnist-flask/archive/master.zip  
Uncompress the zip file. The code must be in a folder called `master`  

### Step 2: Google drive link for mnist data
Download these files into the master folder you just downloaded  
- train.csv : https://drive.google.com/file/d/1-2wqdPvCunWwcenwZwCSYFfFwk38XEXO/view?usp=sharing  
- test.csv  : https://drive.google.com/file/d/1-DVETsf0kvowpAx8H6EDD7ZYJO8cPPRK/view?usp=sharing  

### Step 3: Install requirements
Run this command in the windows powershell
```sh
pip install scikit-image flask pandas scikit-learn
```

### Step 4: Complete the missing code
- Complete missing code in the file named LogisticRegression/train.py
- In the `master` folder, Shift + right-click and open powershell. Run the following command
```sh
python LogisticRegression/train.py --savefile SavedModels/logreg.pkl
```

### Step 5: Start the flask app
- In the `master` folder, Shift + right-click and open powershell. Run the following command
```sh
python app.py --model LR
```
- Open your favourite browser and go to the link http://localhost:5000
- Draw a digit and hit predict!
- The model you trained will be used to classify :)
- Expect medium performance, this is a simple model
- Way better than trying to write our own code (without ML) to identify patterns.
