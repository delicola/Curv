path="/home/hobert/code/NEES-curv"
python_name='Compare.py'
nettype0 = 'real-world'
nettype1 = 'synthetic'
name0 = 'polbooks'
drop0 = 0.1
drop1 = 0.2
drop2 = 0.3

epoch0 = 10
epoch1 = 100
epoch2 = 150

python $path/$python_name --nettype=$nettype0 --name=$name0 --drop_persent=$drop0 --epoch=$epoch0