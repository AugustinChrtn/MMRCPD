all_colors = {
    'blue':    '#377eb8',
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00',
    'black': '#000000',
    'very_light': 	"#F73027",
    'light': '#E64D83',
    'mid': '#CC66C2',
    'dark': '#9E68CC',
    'very_dark': '#69208E'
}


one_step_environments = ['MAB',
                         'ThreeStates',
                         'FourStates']

multi_model_agents = ['SoftmaxMultiModel',
                      
                      'MMRCPD',
                      'MMRCPDNoReassign',
                      'MMRCPDSemiJensen',
                      'RmaxExploration',
                      'RmaxContext',
                      'RmaxNovelty',
                      'MMLowKL',
                      'MMHighKL',
                      'MMLowHorizon',
                      'MMHighHorizon',
                      'MMHighHorizonLowKL',
                      'MMLowMerging',
                      'MMHighMerging',
                      'MMForget',
                      'MMNoMerging',
                      'Baseline']

mM_and_RLCD = multi_model_agents + ['RLCD']


colors = {'VI_softmax': 'gray',
          'reward': 'blue',

          'SoftmaxFiniteHorizon': 'mid',
          'SoftmaxFiniteHorizon10': 'mid',
          'SoftmaxFiniteHorizon5': 'light',
          'SoftmaxFiniteHorizon20': 'dark',
          'SoftmaxFiniteHorizon3': 'very_light',
          'SoftmaxFiniteHorizon30': 'very_dark',

          'MMRCPD': 'blue',
          'MMRCPDNoReassign': 'red',
          'MMRCPDSemiJensen': 'green',


          'Baseline': 'blue',
          'MMLowKL': 'red',
          'MMHighKL': 'orange',
          'MMLowHorizon': 'gray',
          'MMHighHorizon': 'green',
          'MMHighHorizonLowKL': 'pink',


          'MMLowMerging': 'black',
          'MMHighMerging': 'yellow',
          'MMForget': 'brown',
          'MMNoMerging': 'purple',

          'RLCD': 'green',

          'nb_model': 'blue',
          'nb_creation': 'red',
          'nb_merging': 'gray',
          'nb_forgetting': 'green'}

labels = {'VI_softmax': 'Infinite Horizon',
          'reward': 'Reward',

          'SoftmaxFiniteHorizon': 'Finite Horizon',
          'SoftmaxFiniteHorizon3': 'Finite Horizon h=3',
          'SoftmaxFiniteHorizon5': 'Finite Horizon h=5',
          'SoftmaxFiniteHorizon10': 'Finite Horizon h=10',
          'SoftmaxFiniteHorizon20': 'Finite Horizon h=20',
          'SoftmaxFiniteHorizon30': 'Finite Horizon h=30',

          'MMRCPD': 'MMRCPD',
          'MMRCPDNoReassign': 'MMRCPD No Reassign',
          'MMRCPDSemiJensen': 'MMRCPD Semi Jensen',
          'Baseline': 'Baseline',

          'MMLowKL': 'High Creation (Δc=0.3)',
          'MMHighKL': 'Low Creation (Δc=1.5)',
          'MMLowHorizon': 'Short Horizon (h=3)',
          'MMHighHorizon': 'Long Horizon (h=20)',
          'MMHighHorizonLowKL': 'Long Horizon, High Creation (h=20 Δc=0.3)',


          'MMLowMerging': 'Low Merging (Δm=0.01)',
          'MMHighMerging': 'High Merging (Δm=0.3)',
          'MMForget': 'No Merging, Low Memory (Δm=0, maxMod=2)',
          'MMNoMerging': 'No Merging (Δm=0)',
          'RLCD': 'RL with Context Detection',

          'nb_model': 'Number of models',
          'nb_creation': 'Number of models created',
          'nb_merging': 'Number of models merged',
          'nb_forgetting': 'Number of models deleted'}

markers = {'VI_softmax': 'X',
           'reward': '8',

           'SoftmaxFiniteHorizon': '*',
           'SoftmaxFiniteHorizon3': '*',
           'SoftmaxFiniteHorizon5': '*',
           'SoftmaxFiniteHorizon10': '*',
           'SoftmaxFiniteHorizon20': '*',
           'SoftmaxFiniteHorizon30': '*',

           'MMRCPD': '8',
           'MMRCPDNoReassign': '8',
           'MMRCPDSemiJensen': '8',
           'Baseline': '8',

           'MMLowKL': '*',
           'MMHighKL': '^',
           'MMLowHorizon': '*',
           'MMHighHorizon': '^',
           'MMHighHorizonLowKL': 'X',


           'MMLowMerging': '*',
           'MMHighMerging': '^',
           'MMForget': 'X',
           'MMNoMerging': 'd',

           'RLCD': '^',

           'nb_model': '8',
           'nb_creation': '*',
           'nb_merging': 'X',
           'nb_forgetting': '^'}


smoothing_factors = {}
for agent_name in colors.keys():
    smoothing_factors[agent_name] = None
