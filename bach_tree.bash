
BASEDIR=. # software versions
python kdtree.py 1 > $BASEDIR/1_statistics_2Mx128.log
for i in {0..5}; do     
    python kdtree.py $((10+i*20)) > $BASEDIR/$((10+i*20))_first_statistics_2Mx128.log
done

# python kdtree.py 1 > $BASEDIR/1_first_statistics_2Mx128.log