MAGS_default_domains = {
    'wa': {'PRNR': [-8, 8],
           'NMC': [-5, 2],
           'MIMD': [-5, 15],
           'MBPI': [-5, 25],
           'WAVY': [0, 5000],
           'FT0': [-2, 2],
           'FT1': [-2, 2],
           'FT2': [-2, 2],
           'LS0': [-1.5, 1.5],
           'LS1': [-1.5, 1.5],
           'LS2': [-3, 3],
           'CS0': [0, 50],
           'CS1': [-100, 1000]
           },
    'nsw': {'PRNR': [-5, 10],
            'NMC': [-2, 2],
            'MIMD': [-5, 15],
            'MBPI': [-5, 30],
            'WAVY': [0, 5000],
            'FT0': [-2, 2],
            'FT1': [-2, 2],
            'FT2': [-3, 3],
            'LS0': [-2, 2],
            'LS1': [-2, 2],
            'LS2': [-2, 50],
            'CS0': [0, 50],
            'CS1': [-100, 1000]
            },
    'de': {'PRNR': [-5, 5],
           'NMC': [-2, 2],
           'MIMD': [-2, 2],
           'MBPI': [-5, 15],
           'WAVY': [0, 5000],
           'FT0': [-1, 1],
           'FT1': [-1, 1],
           'FT2': [-1, 1],
           'LS0': [-.5, 2],
           'LS1': [-.5, 2],
           'LS2': [-2, 50],
           'CS0': [0, 50],
           'CS1': [-100, 1000]
           }
}
truth_criteria= {
    'wa': 'high_thresh',
    'nsw': 'three_yes',
    'de': 'three_yes'
}
full_region_names = {
    'de': 'Germany',
    'nsw': 'New South Wales',
    'wa': 'Western Australia'
}
parametric_methods = [
    'PRNR',
    'NMC',
    'MIMD',
    'MBPI',
    'WAVY',
    'FT0',
    'FT1',
    'FT2',
    'LS0',
    'LS1',
    'LS2',
    'CS0',
    'CS1',
    'ALL'
]
lstm_features = [
    'PRNR',
    'NMC',
    'MIMD',
    'MBPI',
    'WAVY',
    'FT0',
    'FT1',
    'LS0',
    'LS1',
    'CS0',
    'CS1',
]
rf_features = [
    'PRNR',
    'NMC',
    'MIMD',
    'MBPI',
    'WAVY',
    'FT0',
    'FT1',
    'FT2',
    'LS0',
    'LS1',
    'LS2',
    'CS0',
    'CS1'
]