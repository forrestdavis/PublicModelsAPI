import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot():
    """
        Plots results from demo experiments using summary table

        Returns: 
            A plot with 4 subplots: 
                1. Errors across all stimuli broken by language
                2. Errors across stimuli grouped by number and language
                3. Errors across stimuli grouped by referents (only sg head
                nouns)
                4. Errors from human studies 
    """
    summary = summarize()

    # Set seaborn scheme 
    sns.set(style="whitegrid", color_codes=True)

    #Round for clarity in figures and add accuracy
    summary['accuracy'] = 1 - summary['error']
    summary = summary.round(2)

    # Parent plot
    f, axes = plt.subplots(figsize=(10, 8), ncols=2, nrows=2)
    sns.despine(left=True)
    f.tight_layout(pad=5.0)

    # Plot of global errors
    globalSubset = summary[(summary['model'] != 'Human') &
                            (summary['group'] == 'All Stimuli')]
    globalAcc = sns.barplot(
                data = globalSubset, 
                x = 'group',
                y = 'accuracy', 
                hue = 'lang',
                errorbar = None, 
                ax=axes[0, 0], 
                palette=['mediumpurple', 'goldenrod'])

    # Add label
    for bars in globalAcc.containers:
        globalAcc.bar_label(bars)

    # Set the legend
    globalAcc.legend(title='', loc='lower left')

    globalAcc.set(xlabel="All Stimuli", ylabel="Agreement Accuracy",
                  xticklabels=[], title= "Neural Model Accuracy")

    # Plot by Number
    numSubset = summary[(summary['model'] != 'Human') &
                            (summary['group'] == 'Subject Number')]

    numAcc = sns.barplot(
            data = numSubset, 
            x = 'num', 
            y = 'accuracy',
            hue = 'lang', 
            errorbar = None, 
            ax = axes[0, 1], 
            palette=['mediumpurple', 'goldenrod'])

    numAcc.legend(title='', loc='lower left')

    numAcc.set(xlabel="Subject Number", ylabel="Agreement Accuracy",
                  xticklabels=['Singular', 'Plural'], 
               title= "Neural Model Accuracy by Subject Number")

    # Set hatch format
    hatches = ["//", "||", "//", "||"]
    # Loop over the bars
    for bars in numAcc.containers:
        # Add label
        numAcc.bar_label(bars)
        # Set a different hatch for each group of bars
        for bar in bars:
            hatch = hatches.pop(0)
            bar.set_hatch(hatch)

    # Plot by Referents
    refSubset = summary[(summary['model'] != 'Human') &
                            (summary['group'] == 'Possible Referents')]
    
    refAcc = sns.barplot(
            data = refSubset, 
            x = 'num',
            y = 'accuracy', 
            hue = 'lang', 
            errorbar = None, 
            ax = axes[1, 0], 
            palette=['mediumpurple', 'goldenrod'])

    refAcc.legend(title='', loc='lower left')

    refAcc.set(xlabel="Possible Referents", ylabel="Agreement Accuracy",
                  xticklabels=['One', 'Multiple'], 
               title= "Neural Model Accuracy by Possible Referents")

    # Set hatch format
    hatches = ["*", ".", "*", "."]
    # Loop over the bars
    for bars in refAcc.containers:
        # Add label
        refAcc.bar_label(bars)
        # Set a different hatch for each bar
        for bar in bars:
            hatch = hatches.pop(0)
            bar.set_hatch(hatch)

    # Plot humans
    humSubset = summary[(summary['model'] == 'Human')]# & 
                        #(summary['group'] != 'All Stimuli')]

    humAcc = sns.barplot(
             data = humSubset, 
             x = 'num', 
             y = 'accuracy', 
             hue = 'lang', 
             errorbar = None, 
            ax = axes[1, 1], 
            palette=['mediumpurple', 'goldenrod'])

    humAcc.legend(title="", loc='lower left')

    humAcc.set(xlabel="Contrasts", ylabel="Agreement Accuracy",
                  xticklabels=['Aggregate', 'Singular', 'Plural', 'One', 'Multiple'], 
               title= "Human Accuracy by Subject Number and Possible Referents")

    hatches = ["", "//", "||", "*", ".", "", "//", "||", "*", "."]
    for bars in humAcc.containers:
        humAcc.bar_label(bars)
        for bar in bars:
            hatch = hatches.pop(0)
            bar.set_hatch(hatch)


    return f, axes

def summarize(en_fname='results/Full_English_gpt2_prob.tsv', es_fname='results/Full_Spanish_DeepESP_gpt2-spanish_prob.tsv'):

    """
        Takes results files for the experiments and creates a summary table in
        addition to adding human results. 

        Args:
            en_fname: name of English results file (default: results from
                        gpt2-small on stimuli/Full_English.tsv)
            es_fname: name of Spanish results file (default: results from
                        DeepESP/gpt2-spanish (available on huggingface) 
                        on stimuli/Full_Spanish.tsv)

        Returns:
            pd.DataFrame: summary dataframe with the following attributes:
                group: Experimental Group (All Stimuli, Subject Number, 
                                      Possible Referents)
                num: Number of head noun (all cases of mismatch; sg, pl, both)
                     or number of refernts (one or multi)
                model: Model (Spanish, English, or Human)
                lang: Language (Spanish or English)
                error: Proportion of errors across relevant split

        Notes: 
            Human results for Spanish and the single vs. multi token English are
            from Experiments 1 and 3, respectively, in 
            Vigliocco et al. (1996). Subject-verb agreement in Spanish 
            and English: Differences in the role of conceptual constraints. 
            DOI: 10.1016/S0010-0277(96)00713-5 

            Human results for English agreement (non-single vs. multi) are from
            Experiment 1 in
            Bock and Miller (1991). Broken Agreement. 
            DOI: 10.1016/0010-0285(91)90003-7

    """
    # Set variable for later
    measure = 'prob' # the type of measurement

    # Load initial Spanish results file
    es_data = pd.read_csv(es_fname, sep='\t')
    # Load initial English results file
    en_data = pd.read_csv(en_fname, sep='\t')

    # Flatten the results for combining two languages
    # 1. For each language
    #   a. Get column(s) with measurement
    #   b. Get non-measurement columns
    #   c. Melt (i.e. flatten) dataframe
    #   d. Rename format model names (which include measurement at first)
    # 2. Combine languages in one dataframe

    # Spanish
    columns = []
    columns = list(filter(lambda x: '_'+measure in x, es_data.columns.to_list()))

    base_columns = list(filter(lambda x: f"_{measure}" not in x, es_data.columns.to_list()))

    es_data = pd.melt(es_data, 
                        id_vars=base_columns, 
                        var_name='model', 
                        value_name=measure)

    models = list(map(lambda x: x.replace('_'+measure, ''),
                      es_data['model'].tolist()))

    es_data['model'] = models

    # Remove gender columns from Spanish data
    es_data = es_data.drop(columns=['gen1', 'gen2'])

    # English
    columns = []
    columns = list(filter(lambda x: '_'+measure in x, en_data.columns.to_list()))

    base_columns = list(filter(lambda x: f"_{measure}" not in x, en_data.columns.to_list()))

    en_data = pd.melt(en_data, 
                        id_vars=base_columns, 
                        var_name='model', 
                        value_name=measure)

    models = list(map(lambda x: x.replace('_'+measure, ''),
                      en_data['model'].tolist()))

    en_data['model'] = models

    # Combine dataframes
    data = pd.concat([es_data, en_data])

    # Mark cases of error (where the ungrammatical target is more likely than 
    #                           the grammatical target, 
    #                           for example: Prob(are|the cat) > Prob(is|the
    #                           cat)

    # Variable to hold error marks
    errors = []
    # Loop through data (row by row, which is slow but makes careful 
    #                   checking easier)
    for _, row in data.iterrows():
        # Pairs group items together and order demarks grammatical (x) 
        #                           or ungrammatical (y)
        if row['order'] == 'y':
            continue

        x = row[measure]

        # Get minimally different row (care with language)
        filtered = data
        filtered = filtered[(filtered['pairs'] == row['pairs']) & 
                           (filtered['order'] == 'y') & 
                           (filtered['lang'] == row['lang'])]


        # Get the probability for corresponding item and check that it's unique
        y =  filtered[measure].tolist()
        assert len(y) == 1

        y = y[0]
        # Returns 1 if grammatical is less likely than ungrammatical
        # Else 0
        errors.append(int(x < y))
        errors.append(int(x < y))

    # Add error variable to data
    data['error'] = errors
    # Remove copies
    data = data[data['order'] == 'x']

    # Create summary dataframe for clarity and to add humans results 
    # from papers

    es_all = data[data['lang']=='Spanish']['error'].mean()
    en_all = data[data['lang']=='English']['error'].mean()

    summary = {'group': ['All Stimuli', 'All Stimuli', 'All Stimuli', 
                         'All Stimuli'], 
               'num': ['both', 'both', 'both', 'both'],
               'model': ['Spanish', 'English', 'Human', 'Human'],
               'lang': ['Spanish', 'English', 'Spanish', 'English'], 
               'error': [es_all, en_all, 0.0781, 0.156]}

    # Break down by number
    es_sg = data[(data['lang'] == 'Spanish') & (data['num1'] ==
                                                'sg')]['error'].mean()
    en_sg = data[(data['lang'] == 'English') & (data['num1'] ==
                                                'sg')]['error'].mean()
    es_pl = data[(data['lang'] == 'Spanish') & (data['num1'] ==
                                                'pl')]['error'].mean()
    en_pl = data[(data['lang'] == 'English') & (data['num1'] ==
                                                'pl')]['error'].mean()

    summary['group'].extend(['Subject Number']*4)
    summary['num'].extend(['sg', 'sg', 'sg', 'sg'])
    summary['model'].extend(['Spanish', 'English', 'Human', 'Human'])
    summary['lang'].extend(['Spanish', 'English', 'Spanish', 'English'])
    summary['error'].extend([es_sg, en_sg, .1152, .1562])

    summary['group'].extend(['Subject Number']*4)
    summary['num'].extend(['pl', 'pl', 'pl', 'pl'])
    summary['model'].extend(['Spanish', 'English', 'Human', 'Human'])
    summary['lang'].extend(['Spanish', 'English', 'Spanish', 'English'])
    summary['error'].extend([es_pl, en_pl, 0.0410, 0.019])

    # Break down by referent

    es_sing = data[(data['lang'] == 'Spanish') & (data['num1'] ==
                                                'sg') & (
                                                    data['type'] == 
                                                    'single')]['error'].mean()
    es_multi = data[(data['lang'] == 'Spanish') & (data['num1'] ==
                                                'sg') & (
                                                    data['type'] == 
                                                    'multi')]['error'].mean()

    en_sing = data[(data['lang'] == 'English') & (data['num1'] ==
                                                'sg') & (
                                                    data['type'] == 
                                                    'single')]['error'].mean()
    en_multi = data[(data['lang'] == 'English') & (data['num1'] ==
                                                'sg') & (
                                                    data['type'] == 
                                                    'multi')]['error'].mean()

    summary['group'].extend(['Possible Referents']*4)
    summary['num'].extend(['one', 'one', 'one', 'one'])
    summary['model'].extend(['Spanish', 'English', 'Human', 'Human'])
    summary['lang'].extend(['Spanish', 'English', 'Spanish', 'English'])
    summary['error'].extend([es_sing, en_sing, 0.0742, 0.038])

    summary['group'].extend(['Possible Referents']*4)
    summary['num'].extend(['multi', 'multi', 'multi', 'multi'])
    summary['model'].extend(['Spanish', 'English', 'Human', 'Human'])
    summary['lang'].extend(['Spanish', 'English', 'Spanish', 'English'])
    summary['error'].extend([es_multi, en_multi, 0.1563, 0.0402])


    # To dataframe
    summary = pd.DataFrame.from_dict(summary)

    return summary

if __name__ == "__main__":
    f, axes = plot()
    plt.show()
