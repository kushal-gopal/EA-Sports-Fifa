def preprocess(df, test=False):
    # Unnecessay columns to drop
    columns_to_drop = ['sofifa_id', 'player_url', 'short_name', 'long_name', 'dob', 'potential', 'real_face', 'loaned_from', 
                       'nation_jersey_number', 'joined', 'nationality', 'club', 'potential', 'player_tags', 
                       'team_jersey_number', 'player_traits', 'nation_position', 'team_position' ]
    if test:
        columns_to_drop.remove('club')
        columns_to_drop.remove('short_name')

    fifa_dropped = df.drop(columns_to_drop, axis=1)

    # find null columns to handle missing values
    null_columns = []
    for col in fifa_dropped.columns:
        if fifa_dropped[col].isna().any():
            null_columns.append(col)
        else:
            pass

    # Import Simple Imputer from sklearn to fill missing data
    from sklearn.impute import SimpleImputer
    # fill missing values for given columns using their mode
    imp = SimpleImputer(strategy='most_frequent')
    mode_cols = ['release_clause_eur', 'contract_valid_until']

    fifa_dropped[mode_cols] = imp.fit_transform(fifa_dropped[mode_cols])

    # fill the missing values of this columns using their mean values
    mean_cols = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    imp_mean = SimpleImputer(strategy='mean')
    fifa_dropped[mean_cols] = imp_mean.fit_transform(fifa_dropped[mean_cols])

    # Fill this column missing values with all 0s
    col_0 = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning']
    imp_0 = SimpleImputer(fill_value=0)
    fifa_dropped[col_0] = imp_0.fit_transform(fifa_dropped[col_0])

    # again fill this columns with mode
    mode_columns = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
       'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb',
       'lcb', 'cb', 'rcb', 'rb']

    mode_imp = SimpleImputer(strategy='most_frequent')
    fifa_dropped[mode_columns] = mode_imp.fit_transform(fifa_dropped[mode_columns])

    # create a function to preprocess some columns
    def extract(s):
        s = s.split('+')[0]
        return s
    for col in mode_columns:
        fifa_dropped[col] = fifa_dropped[col].apply(extract)
        fifa_dropped[col] = fifa_dropped[col].astype(int)


    # Remove player position column
    fifa_dropped.drop(['player_positions'], axis=1, inplace=True)

    # function to make data of some columns useful
    def extract_1(s):
        if s.find('+') != -1:
            return s.split('+')[0]
        if s.find('-'):
            return s.split('-')[0]
        return s


    clean_cols = ['attacking_crossing',
     'attacking_finishing',
     'attacking_heading_accuracy',
     'attacking_short_passing',
     'attacking_volleys',
     'skill_dribbling',
     'skill_curve',
     'skill_fk_accuracy',
     'skill_long_passing',
     'skill_ball_control',
     'movement_acceleration',
     'movement_sprint_speed',
     'movement_agility',
     'movement_reactions',
     'movement_balance',
     'power_shot_power',
     'power_jumping',
     'power_stamina',
     'power_strength',
     'power_long_shots',
     'mentality_aggression',
     'mentality_interceptions',
     'mentality_positioning',
     'mentality_vision',
     'mentality_penalties',
     'mentality_composure',
     'defending_marking',
     'defending_standing_tackle',
     'defending_sliding_tackle',
     'goalkeeping_diving',
     'goalkeeping_handling',
     'goalkeeping_kicking',
     'goalkeeping_positioning',
     'goalkeeping_reflexes']

    if test == False:
        for col in clean_cols:
            fifa_dropped[col] = fifa_dropped[col].apply(extract_1)
            fifa_dropped[col] = fifa_dropped[col].astype(int)
            
    fifa_dropped.drop(['body_type'], axis=1, inplace=True)
    return fifa_dropped