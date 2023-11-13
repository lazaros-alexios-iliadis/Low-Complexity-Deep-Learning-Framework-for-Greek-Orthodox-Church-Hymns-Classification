import splitfolders

# split the dataset into train, validation and test data
splitfolders.ratio("D:\\Projects\\Ymnodos\\AppliedSciences\\Mel\\",
                   output="D:\\Projects\\Ymnodos\\AppliedSciences\\Data\\",
                   ratio=(.7, .15, .15))  # The ratio of split dataset (train, val, test)
