#!/usr/bin/env bash

usage_exit() {
        echo "Usage: $0 ex) dcgan-celeba/v0"
        exit 1
}

value1=$1
version=0

while getopts v:h OPT
do
    case ${OPT} in
        v) version=$OPTARG;;
        h) usage_exit;;
        \?) usage_exit;;
    esac
done

rm -rf ./scripts/check_generated_img/${value1}/v${version}
mkdir ./scripts/check_generated_img/${value1}/v${version}
cp ./checkpoint/${value1}/v${version}/*.png ./scripts/check_generated_img/${value1}/v${version}/

