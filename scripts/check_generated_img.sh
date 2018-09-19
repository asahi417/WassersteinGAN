#!/usr/bin/env bash

usage_exit() {
        echo "Usage: $0 ex) dcgan-celeba"
        exit 1
}

#value1=$1

while getopts :d:v:h OPT
do
    case ${OPT} in
        d) value1=$OPTARG;;
        v) version=$OPTARG;;
        h) usage_exit;;
        \?) usage_exit;;
    esac
done

rm -rf ./scripts/check_generated_img/${value1}/v${version}
mkdir -p ./scripts/check_generated_img/${value1}/v${version}
cp ./checkpoint/${value1}/v${version}/*.png ./scripts/check_generated_img/${value1}/v${version}/

