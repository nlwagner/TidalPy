import pytest

import numpy as np

import TidalPy
TidalPy.test_mode()

from TidalPy.rheology.models import MaxwellRheology
from TidalPy.radial_solver.numerical.solutions import find_num_solutions
from TidalPy.radial_solver.numerical.initial.driver import find_initial_conditions

frequency    = 0.01
radius       = 0.1
density      = 7000.
bulk_modulus = 100.0e9

shear         = 5.0e9
viscosity     = 1.0e20
rheo_inst     = MaxwellRheology()
complex_shear = rheo_inst(frequency, shear, viscosity)

G_to_use = 6.6743e-11

known_results_tpy0p4 = {
    # (True, True, True, True, 2): NotImplementedError()
    # (True, True, True, True, 3): NotImplementedError()
    # (True, True, True, False, 2): NotImplementedError()
    # (True, True, True, False, 3): NotImplementedError()
    (True, True, False, False, 2): np.asarray(
        [(0.0005556579280322691+2.4600838952597283e-13j),
        (596673368.25219+1.6654840021169146j),
        (0.0003889144820080675+6.150209738149282e-14j),
        (66674344.602420196+0.35182235222654884j),
        (-89208.0378076861-0.0002595284148001829j),
        (-4460401.890384307-0.012976420740009145j),
        (0.00024791350053916-4.971244609545442e-13j),
        (-536851939.6807604-1.3538210556883428j),
        (0.00031197837513478985-1.2428111523863566e-13j),
        (43593512.54043696+0.1806832281305941j),
        (74922.32352197183+0.00018809984337161143j),
        (3746116.17609859+0.009404992168580572j),
        (0.2+0j),
        (20000000000+100j),
        (0.1+0j),
        (10000000000+50j),
        (3.9140139449328135e-08+0j),
        (7.828027889865629e-07+0j)]
        ),
    (True, True, False, False, 3): np.asarray(
        [(8.50579628861942e-05+3.775813283012537e-14j),
        (98024895.99843337+0.30558270342155863j),
        (3.7011592577238853e-05+7.551626566025036e-15j),
        (9804637.030895539+0.05204383578088772j),
        (-12946.082497569354-3.815154079015924e-05j),
        (-906225.7748298552-0.0026706078553111467j),
        (4.306703711380586e-05-6.217219533012539e-14j),
        (-64899895.998433195-0.1409342659215587j),
        (2.8613407422761153e-05-1.2434439066025038e-14j),
        (6445362.969104463+0.0272530392191123j),
        (10088.939640426497+2.3865826504444965e-05j),
        (706225.7748298545+0.0016706078553111478j),
        (0.030000000000000006+0j),
        (6000000000+30j),
        (0.010000000000000002+0j),
        (2000000000+10j),
        (5.871020917399221e-09+0j),
        (2.3484083669596885e-07+0j)]
        ),
    (True, True, False, True, 2): np.asarray(
        [(-2.1907889563698113e-14-1.0750930417613893e-22j),
        (-0.09167182886263643-4.635062143608366e-10j),
        (3.4647784829467455e-14-1.6053572821013508e-24j),
        (0.0030063787680503+4.040159830322401e-12j),
        (1.2979462721565134e-05+6.129770245186991e-15j),
        (0.0006489731360782567+3.0648851225934955e-13j),
        (-1.0841398991169423e-14-5.563512961741072e-23j),
        (-0.07027661375574767-3.6690497373813854e-10j),
        (-2.4373498224018788e-14-1.6053572821013479e-24j),
        (-0.004740164632719762-2.950513971765508e-11j),
        (1.0871559755369195e-05-4.40974458579269e-15j),
        (0.0005435779877684597-2.2048722928963447e-13j),
        (20+0j),
        (2000000000000+10000j),
        (10+0j),
        (1000000000000+5000j),
        (3.914013944932813e-06+0j),
        (7.828027889865623e-05+0j)]
        ),
    (True, True, False, True, 3): np.asarray(
        [(-2.825125478389442e-14-1.393078594370823e-22j),
        (-0.16538892283028972-8.518304958187907e-10j),
        (3.3252940499452475e-14-1.146683772929536e-24j),
        (0.0054881096464736686+1.3223091345427736e-11j),
        (2.3484083669596876e-05+0j),
        (0.0016438858568717813+0j),
        (-1.7157422624115175e-14-8.730552118811708e-23j),
        (-0.1557742682918144-8.14157690776207e-10j),
        (-2.591416435270343e-14-1.1466837729295343e-24j),
        (-0.008194283350587375-4.998863981498097e-11j),
        (2.3484083669596876e-05+0j),
        (0.0016438858568717813+0j),
        (30+0j),
        (6000000000000+30000j),
        (10+0j),
        (2000000000000+10000j),
        (5.871020917399219e-06+0j),
        (0.00023484083669596877+0j)]
        ),
    # (True, False, True, False, 2): NotImplementedError()
    # (True, False, True, False, 3): NotImplementedError()
    (True, False, True, True, 2): np.asarray(
        [0j,
        (1.5978057071564784-3.36365241276446e-22j),
        (2.8000000000001125e-12-1.400000000000112e-20j),
        (0.4199999999999889+5.707546226615891e-23j),
        (-0.0002882579581652016+0j),
        (-0.01441289790826008+0j),
        0j,
        (38.48671250454986+0j),
        0j,
        0j,
        (-0.0054981017863642645+0j),
        (-0.2749050893182132+0j),
        (20+0j),
        (2000000000000+10000j),
        (10+0j),
        (1000000000000+5000j),
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (True, False, True, True, 3): np.asarray(
        [0j,
        (2.235611414312876-2.6560252481194873e-22j),
        (2.0000000000000448e-12-1.0000000000000445e-20j),
        (0.49999999999999556+2.274746684520826e-23j),
        (-0.00037651591633040317+0j),
        (-0.026356114143128224+0j),
        0j,
        (38.40451821170627+0j),
        0j,
        0j,
        (-0.005486359744529466+0j),
        (-0.3840451821170626+0j),
        (30+0j),
        (6000000000000+30000j),
        (10+0j),
        (2000000000000+10000j),
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        ),
    (True, False, False, False, 2): np.asarray(
        [(0.01429185582503069+3.293283835652251e-11j),
        (51191668955.5296+134.65348911947427j),
        (0.0038229639562576754+8.233209589130504e-12j),
        (1096889186.8773015+7.954408811125719j),
        (-7415180.249540198-0.019981870713363683j),
        (-370759012.47700995-0.9990935356681843j),
        (0.00042691531426134615-3.639186419732414e-11j),
        (122471407.52934454-133.40202742206563j),
        (0.00035672882856532134-9.097966049331173e-12j),
        (57018648.56959751-2.444296571951296j),
        (-20545.310463202917+0.019302127400482837j),
        (-1027265.5231601484+0.9651063700241419j),
        (0.2+0j),
        (20000000000+100j),
        (0.1+0j),
        (10000000000+50j),
        (-9.60859860550672e-07+0j),
        (-4.921719721101344e-05+0j)]
        ),
    (True, False, False, False, 3): np.asarray(
        [(0.0014146469556474317+3.2026085923979923e-12j),
        (5256830187.912028+14.446627076410996j),
        (0.0003029293911294865+6.405217184795927e-13j),
        (116171756.4517945+0.8370674696508135j),
        (-742320.6156694497-0.002002227620226377j),
        (-51962443.09686148-0.14015593341584637j),
        (6.634468289484994e-05-3.538902771364908e-12j),
        (25417369.63202471-13.598060196842635j),
        (3.3268936578969e-05-7.077805542729881e-13j),
        (8307574.631587707-0.24157434855125245j),
        (-2680.511759461687+0.0019271104317954352j),
        (-187635.82316231844+0.13489773022568047j),
        (0.030000000000000006+0j),
        (6000000000+30j),
        (0.010000000000000002+0j),
        (2000000000+10j),
        (-9.412897908260081e-08+0j),
        (-6.765159163304033e-06+0j)]
        ),
    (True, False, False, True, 2): np.asarray(
        [(-1.6231761295130784e-14-3.5664002198553597e-19j),
        (1.5968675310874842-1.8265752490730624e-06j),
        (2.8003176206708083e-12-7.022113169782663e-21j),
        (0.4184244669710802-3.462519683916545e-08j),
        (-0.00028766697456129647+1.2986419430977466e-11j),
        (-0.014383348728064824+6.493209715488732e-10j),
        (6.851558589852557e-12-3.244472547145184e-19j),
        (35.256058094137245-1.6647178497120174e-06j),
        (1.4120666593475297e-13-7.022113169782129e-21j),
        (0.7063368588754687-2.9966358152541806e-08j),
        (-0.005235364401714001-1.1752067704786054e-11j),
        (-0.26176822008570005-5.876033852393027e-10j),
        (20+0j),
        (2000000000000+10000j),
        (10+0j),
        (1000000000000+5000j),
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (True, False, False, True, 3): np.asarray(
        [(-2.3188090475796656e-14-2.548006596571804e-19j),
        (2.232044303432239-1.8455986096371817e-06j),
        (2.0004536896302976e-12-5.015795121273153e-21j),
        (0.4977946133599792-2.424504167923635e-08j),
        (-0.0003753567747518705+1.2740049976914254e-11j),
        (-0.026274974232630935+8.918034983839978e-10j),
        (4.883548068148005e-12-2.3180033391362617e-19j),
        (35.607878065695246-1.6833195059745772e-06j),
        (1.0063508651649627e-13-5.015795121272941e-21j),
        (0.5135135784439245-2.1866414279461225e-08j),
        (-0.005229244956661168-1.1529390932632235e-11j),
        (-0.36604714696628177-8.070573652842563e-10j),
        (30+0j),
        (6000000000000+30000j),
        (10+0j),
        (2000000000000+10000j),
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        ),
    (False, True, True, True, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, True, True, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    (False, True, True, False, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, True, False, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    (False, True, False, False, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, False, False, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    (False, True, False, True, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, False, True, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    # (False, False, True, False, 2): NotImplementedError()
    # (False, False, True, False, 3): NotImplementedError()
    (False, False, True, True, 2): np.asarray(
        [0j,
        (38.48671250454986+0j),
        (-0.0054981017863642645+0j),
        (-0.2749050893182132+0j),
        (20+0j),
        0j,
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (False, False, True, True, 3): np.asarray(
        [0j,
        (38.40451821170627+0j),
        (-0.005486359744529466+0j),
        (-0.3840451821170626+0j),
        (30+0j),
        0j,
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        ),
    (False, False, False, False, 2): np.asarray(
        [(0.015028125223425852+0j),
        (51098438281.99037+0j),
        (-7299776.897427232+0j),
        (-364988844.8713617+0j),
        (0.2+0j),
        0j,
        (-9.60859860550672e-07+0j),
        (-4.921719721101344e-05+0j)]
        ),
    (False, False, False, False, 3): np.asarray(
        [(0.001486067730055292+0j),
        (5109843828.199042+0j),
        (-729977.6897427232+0j),
        (-51098438.28199063+0j),
        (0.030000000000000006+0j),
        0j,
        (-9.412897908260081e-08+0j),
        (-6.765159163304033e-06+0j)]
        ),
    (False, False, False, True, 2): np.asarray(
        [(7.69734250090999e-12+0j),
        (38.48671250454986+0j),
        (-0.0054981017863642645+0j),
        (-0.2749050893182132+0j),
        (20+0j),
        0j,
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (False, False, False, True, 3): np.asarray(
        [(5.486359744529475e-12+0j),
        (38.40451821170627+0j),
        (-0.005486359744529466+0j),
        (-0.3840451821170626+0j),
        (30+0j),
        0j,
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        )
}

known_results_tpy0p5 = {
    # (True, True, True, True, 2): NotImplementedError()
    # (True, True, True, True, 3): NotImplementedError()
    # (True, True, True, False, 2): NotImplementedError()
    # (True, True, True, False, 3): NotImplementedError()
    (True, True, False, False, 2): np.asarray(
        [(-0.0032974903825663095-3.963755911749528e-05j),
        (175176397.06620377-3963756.3583191438j),
        (-0.0015979632402011285-1.9818779586862884e-05j),
        (-119952714.22660895-1981878.5528268134j),
        (-89208.03780768627-0.0002595284163026111j),
        (-4460401.890384294-0.012976420582417617j),
        (0.001202948522094405-1.823371942697851e-05j),
        (-502589766.7579282-1823373.1159757504j),
        (0.0006874270038427265-9.11685965667499e-06j),
        (101552151.82972601-911685.4692695896j),
        (74922.32352197179+0.00018809984480102444j),
        (3746116.176098582+0.00940499234710177j),
        (0.2+0j),
        (20000000000+100j),
        (0.1+0j),
        (10000000000+50j),
        (3.9140139449328135e-08+0j),
        (7.828027889865629e-07+0j)]
        ),
    (True, True, False, False, 3): np.asarray(
        [(0.00034539153357579207-3.603143852120341e-06j),
        (143923147.44013065-720628.2393153141j),
        (0.00011864906389058345-1.201047952752512e-06j),
        (27674246.969721925-240209.45156554814j),
        (-12946.082497569347-3.815154089029944e-05j),
        (-906225.7748298562-0.002670607841166828j),
        (0.0003748248055338338-3.6278870396464984e-06j),
        (-8705276.436370298-725577.2613614039j),
        (0.00013073521845812774-1.2092956765136795e-06j),
        (29408958.708779845-241858.9892684878j),
        (10088.939640426479+2.386582669590853e-05j),
        (706225.7748298513+0.0016706078900129977j),
        (0.030000000000000006+0j),
        (6000000000+30j),
        (0.010000000000000002+0j),
        (2000000000+10j),
        (5.871020917399221e-09+0j),
        (2.3484083669596885e-07+0j)]
        ),
    (True, True, False, True, 2): np.asarray(
        [(-1.5648492545498633e-14-7.878850546189414e-24j),
        (-0.09886332852275648-3.725678921747139e-11j),
        (2.4748417735333868e-14-5.564075587264665e-23j),
        (0.004622255179283607+8.413198280884849e-12j),
        (1.2979462721565134e-05+3.4050661269507677e-15j),
        (0.0006489731360782567+1.7025330634753837e-13j),
        (-7.743856422263894e-15+8.73885690652915e-24j),
        (-0.06671780828826844+3.2956732550505496e-11j),
        (-1.7409641588584892e-14+5.334739863953517e-23j),
        (-0.005126796039372598-1.142324486303536e-11j),
        (1.0871559755369195e-05-1.6850404675564649e-15j),
        (0.0005435779877684597-8.425202337782324e-14j),
        (20+0j),
        (2000000000000+10000j),
        (10+0j),
        (1000000000000+5000j),
        (3.914013944932813e-06+0j),
        (7.828027889865623e-05+0j)]
        ),
    (True, True, False, True, 3): np.asarray(
        [(-2.1973198165251105e-14-5.908886480538516e-24j),
        (-0.17551198495387255-1.7726604862722952e-11j),
        (2.586339816624068e-14-6.118098184074178e-23j),
        (0.006854869541659181+1.2270109148433365e-11j),
        (2.3484083669596876e-05+0j),
        (0.0016438858568717813+0j),
        (-1.3344662040978377e-14+1.1928964717313449e-23j),
        (-0.14962637658105402+3.578689378862307e-11j),
        (-2.015546116321364e-14+5.939726585066157e-23j),
        (-0.008388877611222674-1.9962448714615987e-11j),
        (2.3484083669596876e-05+0j),
        (0.0016438858568717813+0j),
        (30+0j),
        (6000000000000+30000j),
        (10+0j),
        (2000000000000+10000j),
        (5.871020917399219e-06+0j),
        (0.00023484083669596877+0j)]
        ),
    # (True, False, True, False, 2): NotImplementedError()
    # (True, False, True, False, 3): NotImplementedError()
    (True, False, True, True, 2): np.asarray(
        [0j,
        (1.1178057071564398+1.31445941189473e-16j),
        (2.000000000000048e-12-9.999999780923672e-21j),
        (0.4999999999999952-2.1907656425472467e-17j),
        (-0.0002882579581652016+0j),
        (-0.01441289790826008+0j),
        0j,
        (38.48671250454986+0j),
        0j,
        0j,
        (-0.0054981017863642645+0j),
        (-0.2749050893182132+0j),
        (20+0j),
        (2000000000000+10000j),
        (10+0j),
        (1000000000000+5000j),
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (True, False, True, True, 3): np.asarray(
        [0j,
        (1.7022780809795177-2.361338510132897e-16j),
        (1.55555555555558e-12-7.777777974556111e-21j),
        (0.5444444444444421+1.9677821512310208e-17j),
        (-0.00037651591633040317+0j),
        (-0.026356114143128224+0j),
        0j,
        (38.40451821170627+0j),
        0j,
        0j,
        (-0.005486359744529466+0j),
        (-0.3840451821170626+0j),
        (30+0j),
        (6000000000000+30000j),
        (10+0j),
        (2000000000000+10000j),
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        ),
    (True, False, False, False, 2): np.asarray(
        [(0.01386292894147277-5.3493113709999115e-06j),
        (52231113098.64315-534925.2355572729j),
        (0.005412395233590024-2.674655291117597e-06j),
        (895053370.7896112-267461.1327213773j),
        (-7415180.249540198-0.0005772366234786814j),
        (-370759012.47700995-0.028861831142528157j),
        (0.0004227009379319253-1.2969873035120581e-06j),
        (75384502.62976544-129698.35440713855j),
        (0.0002768458616218017-6.48493650821719e-07j),
        (64585507.63133167-64849.042341480585j),
        (-20545.310463202914-0.00010250669088028421j),
        (-1027265.5231601481-0.005125334536399571j),
        (0.2+0j),
        (20000000000+100j),
        (0.1+0j),
        (10000000000+50j),
        (-9.60859860550672e-07+0j),
        (-4.921719721101344e-05+0j)]
        ),
    (True, False, False, False, 3): np.asarray(
        [(0.0011876761145574477+1.335440610232515e-05j),
        (5331557406.705979+2670882.3511007545j),
        (0.0003273735999539129+4.451468718909331e-06j),
        (91030251.45590195+890294.1934928383j),
        (-742320.6156694497-6.175848490751012e-05j),
        (-51962443.09686148-0.004323094021929706j),
        (6.511970584425254e-05-2.571315974782954e-07j),
        (17226446.959350087-51426.2335671022j),
        (2.6239004843092982e-05-8.571053240665992e-08j),
        (8888070.100201285-17142.06206681154j),
        (-2680.511759461686-1.335870110883606e-05j),
        (-187635.82316231838-0.0009351090761088992j),
        (0.030000000000000006+0j),
        (6000000000+30j),
        (0.010000000000000002+0j),
        (2000000000+10j),
        (-9.412897908260081e-08+0j),
        (-6.765159163304033e-06+0j)]
        ),
    (True, False, False, True, 2): np.asarray(
        [(-1.159411521080682e-14+7.306655342702125e-25j),
        (1.1158855526127458-9.477888715326224e-12j),
        (2.0002268719076782e-12-1.000001525203892e-20j),
        (0.4988973064558255-5.443924533731052e-12j),
        (-0.00028766697456129647+2.9173433592652277e-15j),
        (-0.014383348728064824+1.4586716796326137e-13j),
        (4.893970421323251e-12-2.7503118712100395e-21j),
        (35.623368870825715-1.3176678871103949e-08j),
        (1.0086190423910918e-13-3.157578103765245e-23j),
        (0.5146125181921023+2.2901374456224e-09j),
        (-0.005235364401714001+1.231434382832149e-12j),
        (-0.26176822008570005+6.157171914160744e-11j),
        (20+0j),
        (2000000000000+10000j),
        (10+0j),
        (1000000000000+5000j),
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (True, False, False, True, 3): np.asarray(
        [(-1.803518148117456e-14+1.1376565425232859e-24j),
        (1.697559404398561-2.328699918331621e-11j),
        (1.5559084252679991e-12-7.777800920751321e-21j),
        (0.5427644306956712-8.294183030343074e-12j),
        (-0.0003753567747518705+5.7212922290121575e-15j),
        (-0.026274974232630935+4.00490456030851e-13j),
        (3.798315164115105e-12-2.1343510304036856e-21j),
        (35.79808862343076-1.2026095903643751e-08j),
        (7.827173395727468e-14-2.4547534982187966e-23j),
        (0.4072266232965567+1.8141063762876672e-09j),
        (-0.005229244956661168+1.2049377520530088e-12j),
        (-0.36604714696628177+8.434564264371061e-11j),
        (30+0j),
        (6000000000000+30000j),
        (10+0j),
        (2000000000000+10000j),
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        ),
    (False, True, True, True, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, True, True, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    (False, True, True, False, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, True, False, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    (False, True, False, False, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, False, False, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    (False, True, False, True, 2): np.asarray(
        [(0.010000000000000002+0j),
        (0.2+0j)]
        ),
    (False, True, False, True, 3): np.asarray(
        [(0.0010000000000000002+0j),
        (0.04000000000000001+0j)]
        ),
    # (False, False, True, False, 2): NotImplementedError()
    # (False, False, True, False, 3): NotImplementedError()
    (False, False, True, True, 2): np.asarray(
        [0j,
        (38.48671250454986+0j),
        (-0.0054981017863642645+0j),
        (-0.2749050893182132+0j),
        (20+0j),
        0j,
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (False, False, True, True, 3): np.asarray(
        [0j,
        (38.40451821170627+0j),
        (-0.005486359744529466+0j),
        (-0.3840451821170626+0j),
        (30+0j),
        0j,
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        ),
    (False, False, False, False, 2): np.asarray(
        [(0.015936411630039934+0j),
        (51098438281.9902+0j),
        (-7299776.897427233+0j),
        (-364988844.87136173+0j),
        (0.2+0j),
        0j,
        (-9.60859860550672e-07+0j),
        (-4.921719721101344e-05+0j)]
        ),
    (False, False, False, False, 3): np.asarray(
        [(0.0009632253678940818+0j),
        (5109843828.199047+0j),
        (-729977.6897427232+0j),
        (-51098438.28199063+0j),
        (0.030000000000000006+0j),
        0j,
        (-9.412897908260081e-08+0j),
        (-6.765159163304033e-06+0j)]
        ),
    (False, False, False, True, 2): np.asarray(
        [(5.498101786364264e-12+0j),
        (38.48671250454986-0j),
        (-0.0054981017863642645+0j),
        (-0.2749050893182132+0j),
        (20+0j),
        0j,
        (-9.608598605506719e-05+0j),
        (-0.004921719721101344+0j)]
        ),
    (False, False, False, True, 3): np.asarray(
        [(4.267168690189594e-12+0j),
        (38.40451821170627-0j),
        (-0.005486359744529466+0j),
        (-0.3840451821170626+0j),
        (30+0j),
        0j,
        (-9.412897908260079e-05+0j),
        (-0.006765159163304032+0j)]
        )
}


@pytest.mark.parametrize('is_solid', (True, False))
@pytest.mark.parametrize('is_static', (True, False))
@pytest.mark.parametrize('is_incompressible', (True, False))
@pytest.mark.parametrize('use_kamata', (True, False))
@pytest.mark.parametrize('degree_l', (2, 3))
def test_initial_condition_driver(is_solid, is_static, is_incompressible, use_kamata, degree_l):
 
    num_sols = find_num_solutions(is_solid, is_static, is_incompressible)
    num_ys = num_sols * 2

    # Create array full of nans for the "output" array. If things worked okay then they should no longer be nan.
    initial_condition_array = np.nan * np.ones((num_sols, num_ys), dtype=np.complex128, order='C')
    
    # TODO: Several ICs are not yet implemented.
    if ((not use_kamata) and is_incompressible and (is_solid or ((not is_solid) and (not is_static)))) \
            or (use_kamata and is_static and is_incompressible and is_solid):

        with pytest.raises(NotImplementedError):
            find_initial_conditions(
                is_solid, is_static, is_incompressible, use_kamata,
                frequency, radius, density, bulk_modulus, complex_shear,
                degree_l, G_to_use, initial_condition_array
                )
    else:
        # Assumptions should be fine.
        find_initial_conditions(
            is_solid, is_static, is_incompressible, use_kamata,
            frequency, radius, density, bulk_modulus, complex_shear,
            degree_l, G_to_use, initial_condition_array
            )

        # Make sure all of the array elements were set.
        assert np.all(np.isnan(initial_condition_array) == False)


# NOTE: The below fails since we use some different underlying equations from the pre-built results defined above.
#  if you want to run this test with USE_TPY0p4=True, then you will need to swap out the definitions used in
#   TidalPy.RadialSolver.initial.common.pyx
USE_TPY0p4 = False

@pytest.mark.parametrize('is_solid', (True, False))
@pytest.mark.parametrize('is_static', (True, False))
@pytest.mark.parametrize('is_incompressible', (True, False))
@pytest.mark.parametrize('use_kamata', (True, False))
@pytest.mark.parametrize('degree_l', (2, 3))
def test_initial_condition_accuracy(is_solid, is_static, is_incompressible, use_kamata, degree_l):

    if USE_TPY0p4:
        # Use TidalPy 0.4.0 precalculated results
        known_results = known_results_tpy0p4
    else:
        known_results = known_results_tpy0p5

    # Get precalculated resultTidalPy v0.4.0 result
    if (is_solid, is_static, is_incompressible, use_kamata, degree_l) not in known_results:
        # Skip
        assert True
    else:
        old_tpy_result = known_results[(is_solid, is_static, is_incompressible, use_kamata, degree_l)]

        # Get new result
        num_sols = find_num_solutions(is_solid, is_static, is_incompressible)
        num_ys = num_sols * 2

        # Create array full of nans for the "output" array. If things worked okay then they should no longer be nan.
        initial_condition_array = np.nan * np.ones((num_sols, num_ys), dtype=np.complex128, order='C')

        find_initial_conditions(
            is_solid, is_static, is_incompressible, use_kamata,
            frequency, radius, density, bulk_modulus, complex_shear,
            degree_l, G_to_use, initial_condition_array
            )
        
        assert np.allclose(initial_condition_array.flatten(), old_tpy_result)
