#! /bin/bash
resource_dir=/home/ubuntu-lee/WangXD/2017BOT_TIA/final/mask/
list=`ls $resource_dir`
for file in $list
do
name=`echo $file | awk -F "/" '{print $(NF)}'`
numid=`echo $name | awk -F'.' '{print $1}'`
svg_name=$numid".svg"
png_name=$numid".png"
cairosvg ./$svg_name -f png -o ./$png_name
done
