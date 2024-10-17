path="/home/hobert/code/NEES-curv"
python_name='Compare.py'
nettype0='real-world'
nettype1='synthetic'
name0='polbooks'

drop00=0.0
drop0=0.1
drop1=0.2
drop2=0.3
drop3=0.4
drop4=0.5
drop5=0.6
drop6=0.7
drop7=0.8
drop8=0.9
drop9=1.0

epoch0=10
epoch1=100
epoch2=150

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop0 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop0 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop0 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop1 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop1 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop1 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop2 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop2 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop2 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop3 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop3 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop3 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop4 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop4 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop4 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop5 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop5 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop5 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop6 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop6 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop6 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop7 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop7 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop7 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop8 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop8 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop8 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop9 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop9 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop9 --epoch=$epoch2

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop00 --epoch=$epoch0
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop00 --epoch=$epoch1
python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_percent=$drop00 --epoch=$epoch2

