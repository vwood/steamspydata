# Steam Spy Data

## Estimating Profit

We estimate profit by taking the product of the owners given by Steam Spy and the price. This obviously doesn't account for sales, free keys and so on. Since the number of owners and the price were used to produce the target variable we could not use them as features.

This is a limitation of the data, as including the price point selected would be interesting.

## Generating Features

Examining common words taken from the titles reveals:

| Word  | Count |
|-------|-------|
| the   |   374 |
| of    |   233 |
| vr    |   102 |
| 2     |    63 |
| a     |    62 |
| space |    55 |
| and   |    46 |
|in        | 37 |
|super     | 32 |
| game     | 27 |
|world     | 27 |
|adventure | 26 |
|edition   | 26 |
|simulator | 26 |
|to        | 26 |
|one       | 21 |
|heroes    | 18 |
|island    | 17 |
|for       | 17 |
|lost      | 17 |

So I included 'VR', '2', 'Space', and 'Super' as indicator features if they were present in the title.

## Linear Regression

While many of these features would benefit from transforms to make them fit a normal distribution, the real outcome we want is to know the linear weights of those features, so the only transform is to remove the mean from the number of players, and the mean and median playtimes. This makes the intercept more easily interpreted.

The regression almost wholly relies on the number of Players as feature. The R^2 drops from 0.857 to 0.140 without this feature. This will be due to the high correlation between the number of Players and the number of Owners.

Without 
Players meaned        0.139128
is_RPG                0.856565
is_puzzle             0.856560
is_platformer         0.856390
is_sandbox            0.856662
is_simulation         0.856582
is_strategy           0.856573
is_survival           0.856180
HasScore              0.856597
HasScore>50%          0.856750
DaysSinceRelease      0.856392
VRInTitle             0.856739
2InTitle              0.856656
SpaceInTitle          0.856730
SuperInTitle          0.856739


Avg Playtime meaned            6.8375
Median Playtime meaned          -3.9889
Players meaned                18.5191
is_RPG                    -91339.3672
is_puzzle                -285090.6131
is_platformer            -181062.4905
is_sandbox                132384.9794
is_simulation             -85141.8353
is_strategy               -86374.0631
is_survival               263060.5015
HasScore                  -94658.9344
HasScore>50%               -8945.2930
DaysSinceRelease            -461.2584
VRInTitle                  45168.0215
2InTitle                  126918.9739
SpaceInTitle              -76115.2464
SuperInTitle              -75600.4340
intercept                 460374.0175
