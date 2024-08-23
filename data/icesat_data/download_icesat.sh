#!/bin/bash

username='' 
password=''

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://data.ornldaac.earthdata.nasa.gov/protected/above/Boreal_AGB_Density_ICESat2/data/boreal_agb_202302061675663075_0216_train_data.csv"
    echo
    exit 1
}


echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc

if command -v curl >/dev/null 2>&1; then
  echo "Using curl to download data"
  cat "../links.txt"  | tr -d '\r' | xargs -n 1 curl -LJO --netrc-file "$netrc" -b "$cookiejar" -c "$cookiejar" || exit_with_error "curl failed to download data"
else
  exit_with_error "No program available to download data"
fi