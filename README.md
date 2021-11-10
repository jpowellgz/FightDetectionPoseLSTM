# FightDetectionPoseLSTM
Fight detection using Open Pose and Bi-LSTM  
Thesis project with paper on review stages.  
  
Abstract  
Currently, in the area of computer vision, researchers work on the automatic detection of violent,
dangerous and suspicious situations. Fights between people are among these situations, and in this
work, the aim is to identify them automatically. We present a proposal of a technique for the
classification of videos containing fights. The method is based in the combination of a deep neural
network for human detection known as Open Pose, and the recurrent neural network known as Long
Short Term Memory. Open Pose estimates postures, which we used to calculate vectors that describe
the general movements of people in a video. These vectors are processed through Long Short Term
Memory to classify fights. This method was verified on three labeled datasets, commonly found in
fight detection research: Movie Fight Dataset, Surveillance Camera Fight Dataset and Violence
Detection Dataset. Respectively, we obtained classification accuracy of: 95%, 67% and 81%. The
method we proposed has a similar performance to recent works on fight detection, and it opens
different possibilities for future work, to improve the accuracy of classification and execution times.  

Implemented on Python 3.7. 

Libraries:  
Open Pose 1.6.0 (CPU Release)  
Open CV 4.5.1.48  
Tensorflow 2.4  
Scikit-Learn 0.24.0  

Tested on:
Movie Fights Dataset  
Nievas, E. B., Suarez, O. D., García, G. B., & Sukthankar, R. (2011) Movie Fight Detection
Dataset. Recovered from: http://visilab.etsii.uclm.es/personas/oscar/FightDetection/

Violence Detection Dataset  
Aktı, Ş., Tataroğlu, G.A., Ekenel, H.K. Surveillance Camera Fight Dataset. Recovered from:
https://github.com/sayibet/fight-detection-surv-dataset. Access date: May 11 2021

Surveillance Camera Dataset  
Bianculli, M., Falcionelli, N., Sernani, P., Tomassini, S., Contardo, P., Lombardi,M., Dragoni,
A.F. A dataset for automatic violence detection in videos, Data in Brief 33 (2020).
doi:10.1016/j.dib.2020.106587.

