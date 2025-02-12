all_colors = {
    'blue':    '#377eb8',
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
}

multi_model_agents = ['SoftmaxMultiModel',
                      'RmaxExploration', 'RmaxContext', 'RmaxNovelty']


colors = {'VI_softmax': 'gray',
          'reward': 'blue',
          'SoftmaxFiniteHorizon': 'red',
          'SoftmaxFiniteHorizon2' : 'green',
          'SoftmaxMultiModel': 'blue',
          'RmaxExploration': 'pink',
          'RmaxContext': 'green',
          'RmaxNovelty': 'yellow',
          'ThompsonBernouilli':'blue',
          'ThompsonBernouilliFiniteHorizon':'red',

          'nb_model': 'blue',
          'nb_creation': 'red',
          'nb_merging': 'gray',
          'nb_forgetting': 'green'}

labels = {'VI_softmax': 'Infinite Horizon',
          'reward': 'Reward',
          'SoftmaxFiniteHorizon': 'Finite Horizon',
          'SoftmaxFiniteHorizon2' : 'Finite Horizon h=20',
          'SoftmaxMultiModel': 'Multi Model (MM)',
          'RmaxExploration': 'MM + Nov + Sur',
          'RmaxContext': 'MM + Nov + Sur + Cov',
          'RmaxNovelty': 'MM + Nov',
          'ThompsonBernouilli':'Thompson',
          'ThompsonBernouilliFiniteHorizon':'ThompsonFiniteHorizon',

          'nb_model': 'Number of models',
          'nb_creation': 'Number of models created',
          'nb_merging': 'Number of models merged',
          'nb_forgetting': 'Number of models deleted'}

smoothing_factors = {'VI_softmax': 1e10,
                     'SoftmaxFiniteHorizon': 1e10,
                     'SoftmaxFiniteHorizon2' : 1e10,
                     'SoftmaxMultiModel': 1e10,
                     'RmaxContext': 1e10,
                     'RmaxExploration': 1e10,
                     'RmaxNovelty': 1e10,
                     'ThompsonBernouilli':1e10,
                     'ThompsonBernouilliFiniteHorizon':1e10,

                     'reward': 1e10,

                     'nb_merging': 1e10,
                     'nb_forgetting': 1e10,
                     'nb_creation': 1e10,
                     'nb_model': 1e10}
