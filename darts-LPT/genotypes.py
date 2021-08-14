from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PDARTS_TS_CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('dil_conv_3x3', 2), 
    ('skip_connect', 0), 
    ('dil_conv_3x3', 3), 
    ('skip_connect', 2), 
    ('dil_conv_5x5', 4)], 
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3x3', 0), 
    ('dil_conv_3x3', 1), 
    ('avg_pool_3x3', 0), 
    ('dil_conv_5x5', 2), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 3), 
    ('sep_conv_3x3', 0), 
    ('dil_conv_3x3', 2)], 
    reduce_concat=range(2, 6))

PDARTS_TS_CIFAR100 = Genotype(
    normal=[('sep_conv_3x3', 0), 
    ('sep_conv_5x5', 1), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('skip_connect', 2), 
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1)], 
    normal_concat=range(2, 6), 
    reduce=[('dil_conv_3x3', 0), 
    ('avg_pool_3x3', 1), 
    ('dil_conv_3x3', 1), 
    ('sep_conv_5x5', 2), 
    ('sep_conv_5x5', 1), 
    ('sep_conv_5x5', 2), 
    ('avg_pool_3x3', 0), 
    ('dil_conv_5x5', 2)], 
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR10_NEW = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR100_NEW = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('dil_conv_5x5', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 1),
            ('skip_connect', 0)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_CIFAR100 = Genotype(
    normal=[('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 0),
            ('skip_connect', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 1),
            ('skip_connect', 0)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR100 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('dil_conv_3x3', 1),
            ('skip_connect', 1),
            ('dil_conv_5x5', 0),
            ('skip_connect', 1),
            ('dil_conv_5x5', 0),
            ('skip_connect', 1),
            ('dil_conv_3x3', 0)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('sep_conv_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 0),
            ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

DARTS_CIFAR10_TS_1ST = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_1ST = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_18_V1 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_18_V2 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

MY_DARTS_CIFAR10 = Genotype(
    normal=[('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 2),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 3),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_50 = Genotype(
    normal=[('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1),
     ('sep_conv_3x3', 0), 
     ('sep_conv_3x3', 1), 
     ('sep_conv_3x3', 0), 
     ('skip_connect', 1), 
     ('skip_connect', 0), 
     ('dil_conv_3x3', 1)],
      normal_concat=range(2, 6), 
      reduce=[('dil_conv_3x3', 0), 
      ('sep_conv_3x3', 1),
       ('max_pool_3x3', 0), 
       ('skip_connect', 2), 
       ('skip_connect', 2), 
       ('skip_connect', 3), 
       ('skip_connect', 2), 
       ('skip_connect', 3)], 
       reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_50 = Genotype(
    normal=[('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('skip_connect', 1), 
    ('skip_connect', 0), 
    ('dil_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('dil_conv_3x3', 1)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 1), 
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 0), 
    ('dil_conv_5x5', 2), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 0)], 
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_34 = Genotype(
    normal=[('skip_connect', 0), 
    ('dil_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('dil_conv_5x5', 1), 
    ('skip_connect', 0), 
    ('skip_connect', 1), 
    ('skip_connect', 0), 
    ('skip_connect', 1)], 
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3x3', 0), 
    ('avg_pool_3x3', 1), 
    ('skip_connect', 2), 
    ('avg_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('avg_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('avg_pool_3x3', 0)], 
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_34 = Genotype(
    normal=[('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('dil_conv_3x3', 2), 
    ('sep_conv_3x3', 0), 
    ('dil_conv_5x5', 1), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 1)], 
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3x3', 1), 
    ('skip_connect', 0), 
    ('skip_connect', 2), 
    ('avg_pool_3x3', 1), 
    ('avg_pool_3x3', 1), 
    ('skip_connect', 2), 
    ('skip_connect', 2), 
    ('skip_connect', 3)], 
    reduce_concat=range(2, 6))

# DARTS_CIFAR100 = Genotype(
#     normal=[('skip_connect', 0),
#             ('skip_connect', 1),
#             ('skip_connect', 0),
#             ('skip_connect', 1),
#             ('skip_connect', 0),
#             ('skip_connect', 1),
#             ('skip_connect', 0),
#             ('skip_connect', 1)],
#     normal_concat=range(2, 6),
#     reduce=[('max_pool_3x3', 0),
#             ('dil_conv_3x3', 1),
#             ('skip_connect', 2),
#             ('avg_pool_3x3', 0),
#             ('skip_connect', 2),
#             ('max_pool_3x3', 0),
#             ('skip_connect', 2),
#             ('avg_pool_3x3', 0)],
#     reduce_concat=range(2, 6))
DARTS_CIFAR100_1ST = Genotype(
    normal=[('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100 = Genotype(
    normal=[('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_18 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('skip_connect', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_ES = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('skip_connect', 1),
            ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2),
            ('max_pool_3x3', 0),
            ('sep_conv_5x5', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_18_ES = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 2),
            ('skip_connect', 0)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('skip_connect', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('sep_conv_5x5', 3),
            ('max_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))


DARTS_CIFAR10_ES = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 2),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 2)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_18_ES = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

PDARTS_TS_CIFAR100_GAMMA_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
PDARTS_TS_CIFAR100_GAMMA_3 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
PDARTS_TS_CIFAR100_GAMMA_0_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
PDARTS_TS_CIFAR100_GAMMA_0_5 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

DARTS_TS_18_CIFAR10_GAMMA_0_5 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_GAMMA_0_1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_GAMMA_2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_GAMMA_3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

DARTS_TS_18_CIFAR10_LAMBDA_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_LAMBDA_0_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_LAMBDA_0_5 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_LAMBDA_3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

PDARTS_TS_18_CIFAR100_LAMBDA_3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_LAMBDA_0_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_LAMBDA_0_5 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_LAMBDA_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))


PDARTS_TS_18_CIFAR100_AB_1 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_AB_4 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

DARTS_TS_18_CIFAR10_AB_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_AB_4 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))

PDARTS_TUNED_CIFAR100 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
PDARTS_TUNED_CIFAR10 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
