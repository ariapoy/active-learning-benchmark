rm -rf *.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/sonar_scale -O sonar.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale -O iris.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/wine.scale -O wine.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale -O glass.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale -O heart.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ionosphere_scale -O ionosphere.txt
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale -O breast-cancer.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale -O australian.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale -O diabetes.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle.scale -O vehicle.txt
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer_scale -O german.numer.txt
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing -O phishing.txt

# glass, vehicle, german.numer, phishing: LIBSVM is not the same as the UCI official.
# sonar: XZhan2021 is not the same as the UCI official.
# breast-cancer: Both are not the same as the UCI official

# 2021/10/10 Zhan said the most of datasets downloaded from LIBSVM;
# remaining datasets downloaded from UCI.
