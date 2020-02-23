# Captcha_Recognition_Digit
【Framework】This is a captcha recognition algorithm tailored for 4 length digit recognition. It's really straightforward for new beginner.
It's suitable for python3 and tensorflow both gpu or cpu version.

### Motivation
Once you need to make a solution for some captcha recognition, the first thing you need is the complete datasets (x and label). However, it's almost impossible for you to label a lot of datasets. Thus, here is a solution for this case. Because the captcha may made up by different fonts of digit or english character. You can use a package called 'Captcha' to easily generate a lot of training data (with label). You can give it a random font, figure size, character position, noise background, rotate angle, etc...Once you have this training data which share the samilar distribution with the test data. Then, you can send this data into a CNN network to do supervised machine learning. In this project, it can be divided into 4 steps. 

# =============================

Main.py 

--1.check the environment and make folder 2.generate the training data 3.save the training data as tf recoder 4.training the network 5.test the captcha

# =============================

Alexnet.py 

--1. a poverful CNN network which can help you make prediction on the captcha

=============================

### Input and Output



### Performance



### Further Research



**Thanks for your patience to read here, if you like this job, please give me a star, thanks.**
