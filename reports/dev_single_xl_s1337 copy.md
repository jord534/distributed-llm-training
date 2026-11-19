Remarks:


According to the loss plots, in both single and single_xl, the val and train loss track each other
THis is a good sign

Going from 200 examples to 2000 examples dropped the loss from about 7 ish to about high 4s low 5s.
At the end of the xl run, there appears to be a slight diversion between val and train loss

The dev model itself is tiny 
parameter counts for small vs xl dev model : 34.7 M (same)
transformer layers parameter count: 0.92M, so most of the parameters are embedding (97% are word embedding parameters, the hidden architecture is very small in comparasion)


tokens run through in training small vs xl dev model : 1.4 M vs 21.1 M

FLOP operations small vs xl model : 


The learning rate ends up aroun 10^-10 ish for the xl model which is oh so tiny, is it worth it to change the learning rate lambda ?

