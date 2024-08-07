# Push the latest benchmark run to git under the tag 'latest_benchmark'
# and a daily tag with the date
set -ex

THIS_DIR=$(dirname "$0")

BENCHMARK_DIR=$THIS_DIR/benchmarks


mkdir -p $BENCHMARK_DIR

cp -r $THIS_DIR/output/benchmark/* $BENCHMARK_DIR/

git add $BENCHMARK_DIR

# if there are no changes, push to git
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to commit"
else
    git commit -m "latest benchmark"
fi

# push the changes to git under tag 'latest_benchmark'
git tag -f latest_benchmark
git push -f origin latest_benchmark

# create a daily tag with the date
tagname=benchmark-$(date +%Y-%m-%d)
git tag -f $tagname
git push -f origin $tagname
