ffmpeg -y -r 2 -i chain10-e10-%d.png -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" chain10-dql.gif
ffmpeg -y -r 2 -i chain10-e10-%d.png -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse,crop=480:400:320:0" chain10-dql-network.gif
