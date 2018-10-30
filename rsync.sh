#!/bin/bash

if ! [ -z ${1} ]; then
  inputdir=${1}
else
  inputdir=`pwd`
fi

echo ${inputdir}
prefix="/Users/tgadfort/Documents/"
basedir=${inputdir#$prefix}
echo ${basedir}
basedir=`dirname ${basedir}`
if [ ${basedir} == '.' ]; then
  dropdir="/Users/tgadfort/Dropbox/"
else
  dropdir="/Users/tgadfort/Dropbox/${basedir}"
fi
echo ${basedir}
echo ${inputdir}
echo ${dropdir}

#basedir=${inputdir}
echo "Input:   ${inputdir}"
echo "Dropbox: ${dropdir}"
rsync -av --delete --exclude 'data' --exclude 'axa' --progress ${inputdir} ${dropdir}
