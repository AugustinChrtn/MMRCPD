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
    'black': '#000000'
}

multi_model_agents = ['SoftmaxMultiModel',
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


colors = {'VI_softmax': 'gray',
          'reward': 'blue',

          'SoftmaxFiniteHorizon': 'red',
          'SoftmaxFiniteHorizon10': 'red',
          'SoftmaxFiniteHorizon5': 'orange',
          'SoftmaxFiniteHorizon20': 'green',


          'SoftmaxMultiModel': 'blue',


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


          'RmaxExploration': 'pink',
          'RmaxContext': 'green',
          'RmaxNovelty': 'yellow',
          'ThompsonBernouilli': 'blue',
          'ThompsonBernouilliFiniteHorizon': 'red',

          'nb_model': 'blue',
          'nb_creation': 'red',
          'nb_merging': 'gray',
          'nb_forgetting': 'green'}

labels = {'VI_softmax': 'Infinite Horizon',
          'reward': 'Reward',

          'SoftmaxFiniteHorizon': 'Finite Horizon',
          'SoftmaxFiniteHorizon5': 'Finite Horizon h=5',
          'SoftmaxFiniteHorizon10': 'Finite Horizon h=10',
          'SoftmaxFiniteHorizon20': 'Finite Horizon h=20',


          'SoftmaxMultiModel': 'Multi Model',
          'Baseline': 'Baseline',

          'MMLowKL': 'High Creation',
          'MMHighKL': 'Low Creation',
          'MMLowHorizon': 'Short Horizon',
          'MMHighHorizon': 'Long Horizon',
          'MMHighHorizonLowKL': 'Long Horizon High Creation',


          'MMLowMerging': 'High Creation Low Merging',
          'MMHighMerging': 'High Creation High Merging',
          'MMForget': 'High Creation No Merging Low Memory',
          'MMNoMerging': 'High Creation No Merging',



          'RmaxExploration': 'MM + Nov + Sur',
          'RmaxContext': 'MM + Nov + Sur + Cov',
          'RmaxNovelty': 'MM + Nov',
          'ThompsonBernouilli': 'Thompson',
          'ThompsonBernouilliFiniteHorizon': 'ThompsonFiniteHorizon',

          'nb_model': 'Number of models',
          'nb_creation': 'Number of models created',
          'nb_merging': 'Number of models merged',
          'nb_forgetting': 'Number of models deleted'}

smoothing_factors = {'VI_softmax': 1e10,
                     'SoftmaxFiniteHorizon': 1e10,
                     'SoftmaxFiniteHorizon5': 1e10,
                     'SoftmaxFiniteHorizon10': 1e10,
                     'SoftmaxFiniteHorizon20': 1e10,
                     'SoftmaxMultiModel': 1e10,
                    'Baseline':1e10,
                     'MMLowKL': 1e10,
                     'MMHighKL': 1e10,
                     'MMLowHorizon': 1e10,
                     'MMHighHorizon': 1e10,
                     'MMHighHorizonLowKL': 1e10,
                     'MMLowMerging': 1e10,
                     'MMHighMerging': 1e10,
                     'MMForget': 1e10,
                     'MMNoMerging': 1e10,



                     'RmaxExploration': 'MM + Nov + Sur',
                     'RmaxContext': 'MM + Nov + Sur + Cov',
                     'RmaxNovelty': 'MM + Nov',
                     'ThompsonBernouilli': 'Thompson',
                     'ThompsonBernouilliFiniteHorizon': 'ThompsonFiniteHorizon',




                     'RmaxContext': 1e10,
                     'RmaxExploration': 1e10,
                     'RmaxNovelty': 1e10,
                     'ThompsonBernouilli': 1e10,
                     'ThompsonBernouilliFiniteHorizon': 1e10,

                     'reward': 1e10,

                     'nb_merging': 1e10,
                     'nb_forgetting': 1e10,
                     'nb_creation': 1e10,
                     'nb_model': 1e10}
