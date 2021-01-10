from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np

#####################################################################################################################
#####################################################################################################################

def keras_ou(a, model):
    x = a.copy()
    # use OrdinalEncoder to transform team and opp to numeric value
    ord_enc = OrdinalEncoder()
    x['team_code'] = ord_enc.fit_transform(x[['team']])
    x.team_code = x.team_code.astype(int)
    x['opp_code'] = ord_enc.fit_transform(x[['opp']])
    x.opp_code = x.opp_code.astype(int)
    Codes = x[['week', 'team', 'team_code', 'opp', 'opp_code', 'ou_result', 'over/under', 'team_score', 'opp_score', 'h_or_a']]

    # drop 'Team' and 'Opp' since they are now encoded
    x.drop(['team', 'opp'], axis = 1, inplace = True)
    
    # new df
    xx = x[['opp_def_rushy_avg', 'tm_def_score_avg', 'tm_def_passy_avg', 'tm_def_rushy_avg', 'opp_def_passy_avg',
            'opp_off_rushy_avg', 'opp_off_score_avg', 'tm_pass_att_avg', 'tm_off_rushy_avg', 'opp_def_score_avg',
            'opp_pass_att_avg', 'opp_def_1std_avg', 'tm_off_score_avg', 'tm_off_passy_avg',
            'opp_sacked_avg', 'tm_sacked_avg', 'tm_def_1std_avg', 'tm_off_1stdwn_avg', 'tm_pass_int_avg',
            'tm_pass_td_avg', 'tm_def_to_avg', 'opp_def_to_avg', 'tm_rush_td_avg', 'spread', 
            'opp_rush_td_avg', 'team_code', 'opp_code', 'opp_winning%', 'over/under', 'ou_result']]
    
    # seperate dependent and independents
    XX = xx.iloc[:,:-1].values
    yy = xx.iloc[:,-1].values
    
    # scale with StandardScaler and fit_transform
    XX = StandardScaler().fit_transform(XX)
    
    # make prediction
    YY_pred = model.predict(XX)
       
    # reshape array
    size = YY_pred.shape[0]
    xxx = YY_pred.reshape(size, ) 
    
    # dataframe formatting
    df = pd.DataFrame({'week': x.week, 'home_away': x.h_or_a, 'team': Codes.team, 't_score': x.team_score,
                       'opp': Codes.opp, 'o_score': x.opp_score, 'o/u': x['over/under'], 'actual': yy, 'predicted': xxx})
    
    dfh = df.loc[df['home_away'] == 0] # home teams
    dfa = df.loc[df['home_away'] == 1] # away teams

    week = pd.merge(dfh, dfa, how = 'left', left_on = ['week', 'o/u', 'opp'], right_on = ['week', 'o/u', 'team'])
    week.drop(['home_away_x', 'opp_x', 'home_away_y', 'opp_y'],axis = 1, inplace = True)
    week.predicted_x = (week.predicted_x + week.predicted_y) / 2
    week.predicted_y = 1 - week.predicted_x
    week.rename(columns = {'team_x': 'home', 'team_y': 'away', 'predicted_x': 'pred_over', 'predicted_y': 'pred_under',
                           't_score_x': 'h_score', 't_score_y': 'a_score'}, inplace = True)
    week['prediction'] = ['Over' if week.pred_over[i] > week.pred_under[i]
                          else 'No Prediction' if week.pred_over[i] == week.pred_under[i]
                          else 'Under' for i in range(len(week))]
    week['confidence'] = [abs(week.pred_over[i] - week.pred_under[i]) for i in range(len(week))]
    week['final_total'] = [(week.h_score[i] + week.a_score[i]) for i in range(len(week))]
    week['outcome'] = ['Over' if (week.actual_x[i]==1) 
                       else 'Under' if (week.actual_x[i]==0)
                       else 'Push' if (week.actual_x[i]==2) | (week.prediction[i] == 'No Prediction')
                       else "Game Not Completed" for i in range(len(week))]
    week['correct'] = [1 if week.prediction[i] == week.outcome[i]
                       else None if (week.outcome[i] == 'Push') | (week.prediction[i] == 'No Prediction')
                       else 0 for i in range(len(week))]
    week.drop(['actual_x', 'actual_y', 'o_score_x', 'o_score_y', 'h_score', 'a_score'], axis = 1, inplace = True)
    week = week.sort_values(by = ['home'], ascending = True).reset_index(drop = True)

    return week


#####################################################################################################################
#####################################################################################################################


def keras_spread(a, model):
    x = a.copy()
    # use OrdinalEncoder to transform team and opp to numeric value
    ord_enc = OrdinalEncoder()
    x['team_code'] = ord_enc.fit_transform(x[['team']])
    x.team_code = x.team_code.astype(int)
    x['opp_code'] = ord_enc.fit_transform(x[['opp']])
    x.opp_code = x.opp_code.astype(int)
    Codes = x[['season', 'week', 'team', 'team_code', 'opp', 'opp_code']]

    # drop 'Team' and 'Opp' since they are now encoded
    x.drop(['team', 'opp'], axis = 1, inplace = True)
    
    # new df
    xx = x[['tm_def_rushy_avg', 'opp_def_rushy_avg', 'opp_off_rushy_avg', 'opp_pass_att_avg', 'opp_def_passy_avg',
            'tm_def_score_avg', 'tm_pass_att_avg', 'opp_off_score_avg', 'tm_def_1std_avg', 'opp_def_1std_avg',
            'opp_def_score_avg', 'tm_def_passy_avg', 'tm_off_totyd_avg', 'opp_off_passy_avg', 'tm_off_score_avg',
            'opp_rush_td_avg', 'tm_sacked_avg', 'tm_def_to_avg', 'opp_off_to_avg', 'opp_sacked_avg', 'tm_off_to_avg',
            'tm_rush_td_avg', 'opp_off_1stdwn_avg', 'opp_def_to_avg', 'tm_pass_td_avg', 'spread', 'opp_pass_td_avg',
            'opp_code', 'over/under', 'team_code', 'tm_winning%', 'spread_outcome']]
    
    # seperate dependent and independents
    XX = xx.iloc[:,:-1].values
    yy = xx.iloc[:,-1].values
    
    # scale with StandardScaler and fit_transform
    
    XX = StandardScaler().fit_transform(XX)
    
    # make prediction
    YY_pred = model.predict(XX)
       
    # reshape array
    size = YY_pred.shape[0]
    xxx = YY_pred.reshape(size, ) 
    
    # dataframe formatting
    df = pd.DataFrame({'week': x.week, 'home_away': x.h_or_a, 'team': Codes.team, 'opp': Codes.opp,
                       'spread': x['spread'], 'actual': yy, 'predicted': xxx, })
    dfh = df.loc[df['home_away'] == 0] # home teams
    dfa = df.loc[df['home_away'] == 1] # away teams

    week = pd.merge(dfh, dfa, how = 'left', left_on = ['week', 'opp'],
                    right_on = ['week', 'team'])
    week.drop(['home_away_x', 'opp_x', 'home_away_y', 'opp_y'],axis = 1, inplace = True)
    week.rename(columns = {'team_x': 'home', 'team_y': 'away', 'predicted_x': 'home_cover',
                           'predicted_y': 'away_cover', 'spread_x': 'home_spread', 'spread_y': 'away_spread'},
                inplace = True)
    week['pred_spread'] = ['Home Cover' if week.home_cover[i] > week.away_cover[i] else 'Away Cover'
                          for i in range(len(week))]
    week['confidence'] = [abs(week.home_cover[i] - week.away_cover[i]) for i in range(len(week))]
    week['spread_result'] = ['Home Cover' if(week.actual_x[i]==1) 
                             else 'Away Cover' if(week.actual_x[i]==0)
                             else "Push" if(week.actual_x[i]==2)
                             else "Game Not Completed" for i in range(len(week))]
    week['correct'] = [1 if week.pred_spread[i] == week.spread_result[i]
                       else None if week.spread_result[i] == 'Push'
                       else 0 for i in range(len(week))]
    week.drop(['actual_x', 'actual_y'], axis = 1, inplace = True)
    week = week.sort_values(by = ['home'], ascending = True).reset_index(drop = True)
    
    return week
    
    
#####################################################################################################################
#####################################################################################################################


def keras_winner(a, model):
    x = a.copy()
    # use OrdinalEncoder to transform team and opp to numeric value
    ord_enc = OrdinalEncoder()
    x['team_code'] = ord_enc.fit_transform(x[['team']])
    x.team_code = x.team_code.astype(int)
    x['opp_code'] = ord_enc.fit_transform(x[['opp']])
    x.opp_code = x.opp_code.astype(int)
    Codes = x[['season', 'week', 'team', 'team_code', 'opp', 'opp_code']]

    # drop 'Team' and 'Opp' since they are now encoded
    x.drop(['team', 'opp'], axis = 1, inplace = True)
    
    # new df
    xx = x[['spread', 'opp_def_score_avg', 'tm_off_score_avg', 'tm_def_score_avg', 'opp_off_totyd_avg', 'opp_winning%',
           'opp_off_score_avg', 'tm_off_to_avg', 'opp_off_to_avg', 'tm_def_rushy_avg', 'tm_off_totyd_avg',
           'opp_def_rushy_avg', 'opp_off_rushy_avg', 'tm_winning%', 'opp_def_totyd_avg', 'tm_def_totyd_avg',
           'tm_rush_att_avg', 'opp_pass_att_avg', 'tm_sacked_avg', 'tm_pass_att_avg', 'opp_pass_td_avg',
           'opp_sacked_avg', 'tm_pass_td_avg', 'tm_def_to_avg', 'opp_def_to_avg', 'opp_rush_td_avg', 'result']]
    
    # seperate dependent and independents
    XX = xx.iloc[:,:-1].values
    yy = xx.iloc[:,-1].values
    
    # scale with StandardScaler and fit_transform
    XX = StandardScaler().fit_transform(XX)
    
    # make prediction
    YY_pred = model.predict(XX)
    
    # reshape array
    size = YY_pred.shape[0]
    xxx = YY_pred.reshape(size, ) 
    
    # dataframe formatting
    df = pd.DataFrame({'week': x.week, 'home_away': x.h_or_a, 'team': Codes.team, 
                       'actual': yy, 'predicted': xxx, 'opp': Codes.opp})
    dfh = df.loc[df['home_away'] == 0] # home teams
    dfa = df.loc[df['home_away'] == 1] # away teams

    week = pd.merge(dfh, dfa, how = 'left', left_on = ['week', 'opp'], right_on = ['week', 'team'])
    week.drop(['home_away_x', 'opp_x', 'home_away_y', 'opp_y'],axis = 1, inplace = True)
    week.rename(columns = {'team_x': 'home', 'team_y': 'away', 'predicted_x': 'prob_home_win',
                           'predicted_y': 'prob_away_win'}, inplace = True)
    week['pred_winner'] = ['Home' if week.prob_home_win[i] > week.prob_away_win[i] 
                           else 'No Prediction' if week.prob_home_win[i] == week.prob_away_win[i]
                           else 'Away' for i in range(len(week))]
    week['confidence'] = [abs(week.prob_home_win[i] - week.prob_away_win[i]) for i in range(len(week))]
    week['winner'] = ['Home' if(week.actual_x[i]==1) 
                      else 'Away' if(week.actual_x[i]==0)
                      else "Tie" if(week.actual_x[i]==2)
                      else "Game Not Completed" for i in range(len(week))]
    week['correct'] = [1 if week.pred_winner[i] == week.winner[i]
                       else None if week.winner[i] == 'Tie'
                       else None if week.pred_winner[i] == 'No Prediction'
                       else 0 for i in range(len(week))]
    week.drop(['actual_x', 'actual_y'], axis = 1, inplace = True)
    week = week.sort_values(by = ['home'], ascending = True).reset_index(drop = True)
    
    return week
    

#####################################################################################################################
#####################################################################################################################

def predict_ou(a, z):
    x = a.copy()
    # use OrdinalEncoder to transform team and opp to numeric value
    ord_enc = OrdinalEncoder()
    x['team_code'] = ord_enc.fit_transform(x[['team']])
    x.team_code = x.team_code.astype(int)
    x['opp_code'] = ord_enc.fit_transform(x[['opp']])
    x.opp_code = x.opp_code.astype(int)
    Codes_week = x[['week', 'team', 'team_code', 'opp', 'opp_code', 'ou_result', 
                    'over/under', 'team_score', 'opp_score', 'h_or_a']]
    
    # drop 'Team' and 'Opp' since they are now encoded
    x.drop(['team', 'opp'], axis = 1, inplace = True)
     
    # seperate features and target
    # X is feature matrix
    X = x.drop(['result', 'spread_outcome', 'ou_result', 'team_score', 'opp_score'], axis = 1)

    # y is target
    y = x['ou_result']
    
    # normalize data using StandardScaler
    scaler = StandardScaler()
    names = X.columns
    d = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(d, columns = names)
    
    # top features (all with importance of 0.025 or higher) calculated earlier
    top_feat = ['opp_def_rushy_avg', 'tm_def_score_avg', 'tm_def_passy_avg', 'tm_def_rushy_avg', 'opp_def_passy_avg',
                'opp_off_rushy_avg', 'opp_off_score_avg', 'tm_pass_att_avg', 'tm_off_rushy_avg', 'opp_def_score_avg',
                'opp_pass_att_avg', 'opp_def_1std_avg', 'opp_off_totyd_avg', 'tm_off_score_avg', 'tm_off_passy_avg',
                'opp_sacked_avg', 'tm_sacked_avg', 'tm_def_1std_avg', 'tm_off_1stdwn_avg', 'tm_pass_int_avg',
                'opp_pass_td_avg', 'tm_pass_td_avg', 'tm_def_to_avg', 'over/under', 'opp_def_to_avg', 
                'tm_rush_td_avg', 'spread', 'opp_rush_td_avg', 'team_code', 'opp_code', 'opp_winning%']

    # Set X as features data
    X = scaled_df[top_feat]
    
    # Use selected model to get probabilities for week selected
    predictions_probs = z.predict_proba(X)
    Codes_week = Codes_week.assign(team_prob_under =
                                   [round(predictions_probs[i][0],5) for i in range(len(predictions_probs))])
    Codes_week = Codes_week.assign(team_prob_over = 
                                   [round(predictions_probs[i][1],5) for i in range(len(predictions_probs))])
    
    # Format DF
    df = pd.DataFrame({'week': x.week, 'home_away': x.h_or_a, 'team': Codes_week.team, 't_score': x.team_score,
                       'opp': Codes_week.opp, 'o_score': x.opp_score, 'o/u': x['over/under'], 'actual': x.ou_result,
                       'team_over': Codes_week.team_prob_over, 'team_under': Codes_week.team_prob_under})
    
    dfh = df.loc[df['home_away'] == 0] # home teams
    dfa = df.loc[df['home_away'] == 1] # away teams

    week = pd.merge(dfh, dfa, how = 'left', left_on = ['week', 'opp', 'o/u'], right_on = ['week', 'team', 'o/u'])
    week.drop(['home_away_x', 'opp_x', 'home_away_y', 'opp_y'], axis = 1, inplace = True)
    week.rename(columns = {'team_x': 'home', 'team_y': 'away', 't_score_x': 'h_score', 't_score_y': 'a_score'},
                inplace = True)
    week['pred_over'] = [(week.team_over_x[i] + week.team_over_y[i]) / 2 for i in range(len(week))]
    week['pred_under'] = [(week.team_under_x[i] + week.team_under_y[i]) / 2 for i in range(len(week))]
    week['prediction'] = ['Over' if week.pred_over[i] > week.pred_under[i]
                          else 'No Prediction' if week.pred_over[i] == week.pred_under[i]
                          else 'Under' for i in range(len(week))]
    week['confidence'] = [abs(week.pred_over[i] - week.pred_under[i]) for i in range(len(week))]
    week['final_total'] = [(week.h_score[i] + week.a_score[i]) for i in range(len(week))]
    week['outcome'] = ['Over' if (week.actual_x[i]==1) 
                       else 'Under' if (week.actual_x[i]==0)
                       else 'Push' if (week.actual_x[i]==2) | (week.prediction[i] == 'No Prediction')
                       else "Game Not Completed" for i in range(len(week))]
    week['correct'] = [1 if week.prediction[i] == week.outcome[i]
                       else None if (week.outcome[i] == 'Push') | (week.prediction[i] == 'No Prediction')
                       else 0 for i in range(len(week))]
    week.drop(['actual_x', 'actual_y', 'o_score_x', 'o_score_y', 'team_under_x', 'team_under_y', 'team_over_x',
              'team_over_y', 'h_score', 'a_score'], axis = 1, inplace = True)
    week = week.sort_values(by = ['home'], ascending = True).reset_index(drop = True)
    
    return week
    

#####################################################################################################################
#####################################################################################################################


def predict_spread(a, z):
    x = a.copy()
    # use OrdinalEncoder to transform team and opp to numeric value
    from sklearn.preprocessing import OrdinalEncoder
    ord_enc = OrdinalEncoder()
    x['team_code'] = ord_enc.fit_transform(x[['team']])
    x.team_code = x.team_code.astype(int)
    x['opp_code'] = ord_enc.fit_transform(x[['opp']])
    x.opp_code = x.opp_code.astype(int)
    Codes_week = x[['team', 'team_code', 'opp', 'opp_code', 'spread', 'spread_outcome', 'h_or_a']]
    
    # drop 'Team' and 'Opp' since they are now encoded
    x.drop(['team', 'opp'], axis = 1, inplace = True)
     
    # seperate features and target
    # X is feature matrix
    X = x.drop(['result', 'spread_outcome', 'ou_result', 'team_score', 'opp_score'], axis = 1)

    # y is target
    y = x['spread_outcome']
    
    # normalize data using StandadScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    names = X.columns
    d = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(d, columns = names)
    
    # top features calculated during feature selection
    top_feat = ['tm_def_rushy_avg', 'opp_def_rushy_avg', 'opp_off_rushy_avg', 'tm_off_rushy_avg', 'tm_def_passy_avg',
                'opp_def_totyd_avg', 'tm_pass_att_avg', 'opp_cmp_avg', 'tm_off_totyd_avg', 'tm_def_1std_avg',
                'opp_def_score_avg', 'opp_off_score_avg', 'tm_def_score_avg', 'tm_sacked_avg', 'tm_off_score_avg',
                'opp_off_totyd_avg', 'opp_sacked_avg', 'opp_def_to_avg', 'tm_def_to_avg', 'spread', 'opp_rush_td_avg',
                'opp_off_to_avg', 'opp_pass_td_avg', 'tm_rush_td_avg', 'team_code', 'over/under', 'tm_pass_td_avg',
                'opp_code', 'tm_winning%']

    # Set X as features data
    X = scaled_df[top_feat]
      
    # Use selected model to get probabilities for week selected
    predictions_probs = z.predict_proba(X)
    Codes_week = Codes_week.assign(team_prob_no_cover =
                                   [round(predictions_probs[i][0],5) for i in range(len(predictions_probs))])
    Codes_week = Codes_week.assign(team_prob_cover = 
                                   [round(predictions_probs[i][1],5) for i in range(len(predictions_probs))])
    
    # Format DF
    df = pd.DataFrame({'week': x.week, 'home_away': x.h_or_a, 'team': Codes_week.team, 't_score': x.team_score,
                       'opp': Codes_week.opp, 'spread': x.spread, 'actual': x.spread_outcome,
                       'predicted': Codes_week.team_prob_cover})
   
    dfh = df.loc[df['home_away'] == 0] # home teams
    dfa = df.loc[df['home_away'] == 1] # away teams

    week = pd.merge(dfh, dfa, how = 'left', left_on = ['week', 'opp'], right_on = ['week', 'team'])
    week.drop(['home_away_x', 'opp_x', 'home_away_y', 'opp_y'], axis = 1, inplace = True)
    week.rename(columns = {'team_x': 'home', 'team_y': 'away', 'predicted_x': 'home_cover',
                           'predicted_y': 'away_cover', 'spread_x': 'home_spread', 'spread_y': 'away_spread'},
                inplace = True)
    week['pred_spread'] = ['Home Cover' if week.home_cover[i] > week.away_cover[i] 
                           else 'Away Cover' if week.home_cover[i] < week.away_cover[i]
                           else 'No Prediction' for i in range(len(week))]
    week['confidence'] = [abs(week.home_cover[i] - week.away_cover[i]) for i in range(len(week))]
    week['spread_result'] = ['Home Cover' if (week.actual_x[i]==1) 
                             else 'Away Cover' if (week.actual_x[i]==0)
                             else 'Push' if (week.actual_x[i]==2) | (week.pred_spread[i] == 'No Prediction')
                             else "Game Not Completed" for i in range(len(week))]
    week['correct'] = [1 if week.pred_spread[i] == week.spread_result[i]
                       else None if (week.spread_result[i] == 'Push') | (week.pred_spread[i] == 'No Prediction')
                       else 0 for i in range(len(week))]
    week.drop(['actual_x', 'actual_y', 't_score_x', 't_score_y'], axis = 1, inplace = True)
    week = week.sort_values(by = ['home'], ascending = True).reset_index(drop = True)
    
    return week
    

#####################################################################################################################
#####################################################################################################################


def predict_winners(a, z):
    x = a.copy()
    # use OrdinalEncoder to transform team and opp to numeric value
    ord_enc = OrdinalEncoder()
    x['team_code'] = ord_enc.fit_transform(x[['team']])
    x.team_code = x.team_code.astype(int)
    x['opp_code'] = ord_enc.fit_transform(x[['opp']])
    x.opp_code = x.opp_code.astype(int)
    Codes_week = x[['week', 'team', 'team_code', 'opp', 'opp_code', 'result']]
    
    # drop 'Team' and 'Opp' since they are now encoded
    x.drop(['team', 'opp'], axis = 1, inplace = True)
    
    # seperate features and target
    # X is feature matrix
    X = x.drop(['result', 'spread_outcome', 'ou_result', 'team_score', 'opp_score'], axis = 1)

    # y is target
    y = x['result']
    
    # normalize data using StandardScaler
    scaler = StandardScaler()
    names = X.columns
    d = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(d, columns = names)
    
    # top features (all with importance of 0.025 or higher) calculated earlier
    top_feat = ['spread', 'opp_def_score_avg', 'tm_off_score_avg', 'tm_def_score_avg', 'opp_off_totyd_avg', 
                'opp_winning%', 'opp_off_score_avg', 'tm_off_to_avg', 'opp_off_to_avg', 'tm_def_rushy_avg', 
                'tm_off_totyd_avg', 'opp_def_rushy_avg', 'tm_winning%', 'opp_def_totyd_avg', 'tm_def_totyd_avg',
                'opp_pass_att_avg', 'tm_sacked_avg', 'tm_off_rushy_avg', 'tm_pass_att_avg', 'opp_pass_td_avg',
                'opp_sacked_avg', 'tm_pass_td_avg', 'tm_def_to_avg', 'opp_def_to_avg', 'opp_rush_td_avg']

    # Set X as features data
    X = scaled_df[top_feat]
    
    # Use selected model to get probabilities for week selected
    predictions_probs = z.predict_proba(X)
    Codes_week = Codes_week.assign(team_prob_loss =
                                   [round(predictions_probs[i][0],3) for i in range(len(predictions_probs))])
    Codes_week = Codes_week.assign(team_prob_win = 
                                   [round(predictions_probs[i][1],3) for i in range(len(predictions_probs))])
    
    # Format DF
    Codes_week.drop(['team_code', 'opp_code'], axis = 1, inplace = True)
    Codes_week = Codes_week[['week', 'team', 'team_prob_win', 'team_prob_loss', 'opp', 'result']]

    df = pd.DataFrame({'week': x.week, 'home_away': x.h_or_a, 'team': Codes_week.team, 'opp': Codes_week.opp,
                       'actual': x.result, 'predicted': Codes_week.team_prob_win})
   
    dfh = df.loc[df['home_away'] == 0] # home teams
    dfa = df.loc[df['home_away'] == 1] # away teams

    week = pd.merge(dfh, dfa, how = 'left', left_on = ['week', 'opp'], right_on = ['week', 'team'])
    week.drop(['home_away_x', 'opp_x', 'home_away_y', 'opp_y'], axis = 1, inplace = True)
    week.rename(columns = {'team_x': 'home', 'team_y': 'away', 'predicted_x': 'prob_home_win',
                           'predicted_y': 'prob_away_win'},
                inplace = True)
    week['pred_winner'] = ['Home' if week.prob_home_win[i] > week.prob_away_win[i] 
                           else 'No Prediction' if week.prob_home_win[i] == week.prob_away_win[i]
                           else 'Away' for i in range(len(week))]
    week['confidence'] = [abs(week.prob_home_win[i] - week.prob_away_win[i]) for i in range(len(week))]
    week['winner'] = ['Home' if(week.actual_x[i]==1) 
                      else 'Away' if(week.actual_x[i]==0)
                      else "Tie" if(week.actual_x[i]==2)
                      else "Game Not Completed" for i in range(len(week))]
    week['correct'] = [1 if week.pred_winner[i] == week.winner[i]
                       else None if week.winner[i] == 'Tie'
                       else None if week.pred_winner[i] == 'No Prediction'
                       else 0 for i in range(len(week))]
    week.drop(['actual_x', 'actual_y'], axis = 1, inplace = True)
    week = week.sort_values(by = ['home'], ascending = True).reset_index(drop = True)

    return week