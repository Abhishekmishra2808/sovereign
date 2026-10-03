#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential git gfortran wget meson ninja-build

mkdir -p /cutest
cd /cutest
git clone --depth 1 https://github.com/ralna/ARCHDefs archdefs
git clone --depth 1 https://github.com/ralna/SIFDecode sifdecode
git clone --depth 1 https://github.com/ralna/CUTEst cutest
git clone --depth 1 https://bitbucket.org/optrove/sif mastsif

export ARCHDEFS=/cutest/archdefs/
export SIFDECODE=/cutest/sifdecode/
export MASTSIF=/cutest/mastsif/
export CUTEST=/cutest/cutest/
export MYARCH=pc64.lnx.gfo

wget -q https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh \
  -O /cutest/.install_cutest.sh
chmod +x /cutest/.install_cutest.sh
/cutest/.install_cutest.sh

pip install pycutest
cp /workspace/benchmarks/data/LISWET1.SIF /cutest/mastsif/LISWET1.SIF
export PATH=/cutest/sifdecode/bin:/cutest/cutest/bin:$PATH
python /workspace/benchmarks/tools/verify_liswet_pycutest.py
