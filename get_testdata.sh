#!/bin/sh
# flake8: noqa

# Pulled from https://github.com/chatcannon/galvani/blob/master/get_testdata.sh

## Test data are posted on FigShare, listed in this article
# http://figshare.com/articles/galvani_test_data/1228760

mkdir -p tests/testdata/aps-txm-data
mkdir -p tests/testdata/ssrl-txm-data
cd tests/testdata

BIN="/usr/bin/wget -nv"

# https://ndownloader.figshare.com/files/3670071

$BIN https://ndownloader.figshare.com/files/3670071 -O NEI-C10_4cycles_C14.mpt
$BIN https://ndownloader.figshare.com/files/3670065 -O NEI-C10_4cycles_C14.mpr
$BIN https://ndownloader.figshare.com/files/3670989 -O ssrl-txm-data/rep01_000001_ref_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_001of001.xrm
$BIN https://ndownloader.figshare.com/files/3671022 -O ssrl-txm-data/rep01_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_001of002.xrm
$BIN https://ndownloader.figshare.com/files/3671025 -O ssrl-txm-data/rep01_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_002of002.xrm
$BIN https://ndownloader.figshare.com/files/3671037 -O aps-txm-data/20151111_UIC_XANES00_sam01_8313.xrm
$BIN https://ndownloader.figshare.com/files/3696405?private_link=10522a3690faac7dfb4d -O NCA-cell2-soc1-fov1-xanesspectrum.tsv
$BIN https://ndownloader.figshare.com/files/3696444?private_link=8c919b5a2fbff2174a9c -O corundum.xye
