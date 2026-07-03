---
description: The literature behind TSDynamics — the original papers for every built-in system, and the method papers behind the analysis toolkit, with DOI links.
---

<span class="ts-kicker">References · Bibliography</span>

# Bibliography

Every built-in system and every analysis method in TSDynamics traces back to a
primary source. This page collects them all — never a competitor library, always
the original paper. It has two halves:

- The **[systems bibliography](#systems)** is generated straight from the
  catalogue. Each built-in system carries a `reference` (and, where one exists, a
  `doi`) class attribute; this page walks the [registry](../reference/registry.md),
  deduplicates, and lists the citing systems under each paper. Add or edit a
  system and its citation lands here automatically.
- The **[methods bibliography](#methods)** collects the papers behind the
  analysis toolkit — the same citations that close each page in the
  [Analysis](../analysis/index.md) section.

!!! tip "Reproducing the numbers"
    The parameter defaults and initial conditions in each system's page match the
    cited paper wherever the source gives them. When a quantity is quoted in the
    docs (a Lyapunov spectrum, a fractal dimension) it is pinned to a fixed
    initial condition and, for stochastic systems, a fixed seed — so every number
    is reproducible from the snippet that produces it.

## Systems {#systems}

The 154 built-in systems cite **124 distinct sources**, grouped below by family and ordered alphabetically by first author. A handful of textbook/folklore systems (the Lissajous figures, a couple of purely illustrative maps) carry no single primary source and are omitted from this list.

### Ordinary differential equations

- Abooee, Yaghini-Bonabi & Jahed-Motlagh (2013), Commun. Nonlinear Sci. Numer. Simul. 18, 1235-1245. [doi:10.1016/j.cnsns.2012.08.036](https://doi.org/10.1016/j.cnsns.2012.08.036)<br><span class="ts-cite-systems">Cited by: `Laser`</span>
- Aizawa & Uezu (1982), Prog. Theor. Phys. 67, 982-985. [doi:10.1143/PTP.67.982](https://doi.org/10.1143/PTP.67.982)<br><span class="ts-cite-systems">Cited by: `Aizawa`</span>
- Anishchenko et al. (2007), Nonlinear Dynamics of Chaotic and Stochastic Systems. [doi:10.1007/978-3-540-38168-6](https://doi.org/10.1007/978-3-540-38168-6)<br><span class="ts-cite-systems">Cited by: `AnishchenkoAstakhov`</span>
- Arena, Caponetto, Fortuna & Porto (1998), Int. J. Bifurc. Chaos 8, 1527. [doi:10.1142/s0218127498001170](https://doi.org/10.1142/s0218127498001170)<br><span class="ts-cite-systems">Cited by: `CellularNeuralNetwork`</span>
- Arneodo, Coullet & Tresser (1980), Phys. Lett. A 79, 259-263. [doi:10.1016/0375-9601(80)90342-4](https://doi.org/10.1016/0375-9601(80)90342-4)<br><span class="ts-cite-systems">Cited by: `Arneodo`</span>
- Arnold (1966), J. Appl. Math. Mech. 30, 223-226. [doi:10.1016/0021-8928(66)90070-0](https://doi.org/10.1016/0021-8928(66)90070-0)<br><span class="ts-cite-systems">Cited by: `ArnoldBeltramiChildress`</span>
- Awrejcewicz & Holicke (1999), Int. J. Bifurc. Chaos. [doi:10.1142/s0218127499000341](https://doi.org/10.1142/s0218127499000341)<br><span class="ts-cite-systems">Cited by: `StickSlipOscillator`</span>
- Bao & Liu (2008), Chin. Phys. Lett. 25, 2396-2399. [doi:10.1088/0256-307x/25/7/018](https://doi.org/10.1088/0256-307x/25/7/018)<br><span class="ts-cite-systems">Cited by: `HyperBao`</span>
- Baran & Raduta (1998), Int. J. Mod. Phys. E. [doi:10.1142/s0218301398000282](https://doi.org/10.1142/s0218301398000282)<br><span class="ts-cite-systems">Cited by: `NuclearQuadrupole`</span>
- Blasius, Huppert & Stone (1999), Nature 399, 354-359. [doi:10.1038/20676](https://doi.org/10.1038/20676)<br><span class="ts-cite-systems">Cited by: `Blasius`</span>
- Bouali (1999), Int. J. Bifurcation Chaos 9, 745-756. [doi:10.1142/s0218127499000535](https://doi.org/10.1142/s0218127499000535)<br><span class="ts-cite-systems">Cited by: `Bouali2`</span>
- Cai & Huang (2007), Int. J. Nonlinear Sci..<br><span class="ts-cite-systems">Cited by: `HyperCai`</span>
- Cai & Huang (2007), Int. J. Nonlinear Sci. 3, 235-241.<br><span class="ts-cite-systems">Cited by: `Finance`</span>
- Chay (1985), Physica D 16, 233-242. [doi:10.1016/0167-2789(85)90060-0](https://doi.org/10.1016/0167-2789(85)90060-0)<br><span class="ts-cite-systems">Cited by: `ExcitableCell`</span>
- Chen & Ueta (1999), Int. J. Bifurcation Chaos 9, 1465-1466. [doi:10.1142/s0218127499001024](https://doi.org/10.1142/s0218127499001024)<br><span class="ts-cite-systems">Cited by: `Chen`</span>
- Chen & Lee (2004), Chaos Solitons Fractals 21, 957-965. [doi:10.1016/j.chaos.2003.12.034](https://doi.org/10.1016/j.chaos.2003.12.034)<br><span class="ts-cite-systems">Cited by: `ChenLee`</span>
- Chen, Lu, Lü & Yu (2006), Physica A 364, 103. [doi:10.1016/j.physa.2005.09.039](https://doi.org/10.1016/j.physa.2005.09.039)<br><span class="ts-cite-systems">Cited by: `HyperLu`</span>
- Dadras & Momeni (2009), Phys. Lett. A 373, 3637-3642. [doi:10.1016/j.physleta.2009.07.088](https://doi.org/10.1016/j.physleta.2009.07.088)<br><span class="ts-cite-systems">Cited by: `Dadras`</span>
- Decroly & Goldbeter (1982), Proc. Natl. Acad. Sci. U.S.A. 79, 6917-6921. [doi:10.1073/pnas.79.22.6917](https://doi.org/10.1073/pnas.79.22.6917)<br><span class="ts-cite-systems">Cited by: `GlycolyticOscillation`</span>
- Duffing (1918), Erzwungene Schwingungen bei veränderlicher Eigenfrequenz, Vieweg, Braunschweig.<br><span class="ts-cite-systems">Cited by: `Duffing`</span>
- Field & Noyes (1974), J. Chem. Phys. 60, 1877-1884. [doi:10.1063/1.1681288](https://doi.org/10.1063/1.1681288)<br><span class="ts-cite-systems">Cited by: `Oregonator`</span>
- FitzHugh (1961), Biophys. J. 1, 445-466. [doi:10.1016/s0006-3495(61)86902-6](https://doi.org/10.1016/s0006-3495(61)86902-6)<br><span class="ts-cite-systems">Cited by: `ForcedFitzHughNagumo`</span>
- Froeschlé, Guzzo & Lega (2000), Science 289, 2108. [doi:10.1126/science.289.5487.2108](https://doi.org/10.1126/science.289.5487.2108)<br><span class="ts-cite-systems">Cited by: `ArnoldWeb`</span>
- Gilet & Bush (2009), J. Fluid Mech. 625, 167-203. [doi:10.1017/s0022112008005442](https://doi.org/10.1017/s0022112008005442)<br><span class="ts-cite-systems">Cited by: `FluidTrampoline`</span>
- Gilpin & Feldman (2017), PLoS Comput. Biol. 13, e1005644. [doi:10.1371/journal.pcbi.1005644](https://doi.org/10.1371/journal.pcbi.1005644)<br><span class="ts-cite-systems">Cited by: `CoevolvingPredatorPrey`</span>
- Guckenheimer & Holmes (1988), Math. Proc. Camb. Phil. Soc. 103, 189-192. [doi:10.1017/s0305004100064732](https://doi.org/10.1017/s0305004100064732)<br><span class="ts-cite-systems">Cited by: `GuckenheimerHolmes`</span>
- Hastings & Powell (1991), Ecology 72, 896-903. [doi:10.2307/1940591](https://doi.org/10.2307/1940591)<br><span class="ts-cite-systems">Cited by: `HastingsPowell`</span>
- Hindmarsh & Rose (1984), Proc. R. Soc. Lond. B 221, 87-102. [doi:10.1098/rspb.1984.0024](https://doi.org/10.1098/rspb.1984.0024)<br><span class="ts-cite-systems">Cited by: `HindmarshRose`</span>
- Houart, Dupont & Goldbeter (1999), Bull. Math. Biol. 61, 507-530. [doi:10.1006/bulm.1999.0095](https://doi.org/10.1006/bulm.1999.0095)<br><span class="ts-cite-systems">Cited by: `CaTwoPlus`</span>
- Hénon & Heiles (1964), Astron. J. 69, 73-79. [doi:10.1086/109234](https://doi.org/10.1086/109234)<br><span class="ts-cite-systems">Cited by: `HenonHeiles`</span>
- Itik & Banks (2010), Int. J. Bifurcation Chaos 20, 71-79. [doi:10.1142/s0218127410025417](https://doi.org/10.1142/s0218127410025417)<br><span class="ts-cite-systems">Cited by: `ItikBanksTumor`</span>
- Kennedy (1994), IEEE Trans. Circuits Syst. I 41, 771-774. [doi:10.1109/81.331536](https://doi.org/10.1109/81.331536)<br><span class="ts-cite-systems">Cited by: `Colpitts`</span>
- Kuramoto & Tsuzuki (1976), Prog. Theor. Phys. 55, 356-369; Sivashinsky (1977), Acta Astronaut. 4, 1177-1206. [doi:10.1143/ptp.55.356](https://doi.org/10.1143/ptp.55.356)<br><span class="ts-cite-systems">Cited by: `KuramotoSivashinsky`</span>
- Leipnik & Newton (1981), Phys. Lett. A 86, 63. [doi:10.1016/0375-9601(81)90165-1](https://doi.org/10.1016/0375-9601(81)90165-1)<br><span class="ts-cite-systems">Cited by: `NewtonLiepnik`</span>
- Leloup, Gonze & Goldbeter (1999); Gonze, Leloup & Goldbeter (2000). [doi:10.1177/074873099129000948](https://doi.org/10.1177/074873099129000948)<br><span class="ts-cite-systems">Cited by: `CircadianRhythm`</span>
- Letellier & Rössler (2007), Scholarpedia 2(8), 1936. [doi:10.4249/scholarpedia.1936](https://doi.org/10.4249/scholarpedia.1936)<br><span class="ts-cite-systems">Cited by: `HyperXu`</span>
- Li (2008), Phys. Lett. A 372, 387-393. [doi:10.1016/j.physleta.2007.07.045](https://doi.org/10.1016/j.physleta.2007.07.045)<br><span class="ts-cite-systems">Cited by: `DequanLi`</span>
- Li et al. (2015), IEICE Electron. Express 12(4), 20141116. [doi:10.1587/elex.12.20141116](https://doi.org/10.1587/elex.12.20141116)<br><span class="ts-cite-systems">Cited by: `Sakarya`</span>
- Lorenz (1963), J. Atmos. Sci. 20, 130-141. [doi:10.1175/1520-0469(1963)020&lt;0130:dnf&gt;2.0.co;2](https://doi.org/10.1175/1520-0469(1963)020%3C0130:dnf%3E2.0.co;2)<br><span class="ts-cite-systems">Cited by: `Lorenz`, `LorenzCoupled`</span>
- Lorenz (1984), 'Irregularity: a fundamental property of the atmosphere', Tellus 36A, 98-110. [doi:10.1111/j.1600-0870.1984.tb00230.x](https://doi.org/10.1111/j.1600-0870.1984.tb00230.x)<br><span class="ts-cite-systems">Cited by: `Hadley`</span>
- Lorenz (1984), Tellus 36A, 98-110. [doi:10.3402/tellusa.v36i2.11473](https://doi.org/10.3402/tellusa.v36i2.11473)<br><span class="ts-cite-systems">Cited by: `Lorenz84`</span>
- Lorenz (1996), Proc. ECMWF Seminar on Predictability 1, 1-18.<br><span class="ts-cite-systems">Cited by: `Lorenz96`</span>
- Lü & Chen (2002), Int. J. Bifurcation Chaos 12, 659-661. [doi:10.1142/s0218127402004620](https://doi.org/10.1142/s0218127402004620)<br><span class="ts-cite-systems">Cited by: `LuChen`</span>
- Lü, Chen, Cheng & Čelikovský (2002), Int. J. Bifurcation Chaos 12, 2917-2926. [doi:10.1142/s021812740200631x](https://doi.org/10.1142/s021812740200631x)<br><span class="ts-cite-systems">Cited by: `LuChenCheng`</span>
- Marion (1965), Classical Dynamics of Particles and Systems, Academic Press. [doi:10.1016/c2013-0-12598-6](https://doi.org/10.1016/c2013-0-12598-6)<br><span class="ts-cite-systems">Cited by: `DoublePendulum`</span>
- Matsumoto (1984), IEEE Trans. Circuits Syst. 31, 1055-1058. [doi:10.1109/tcs.1984.1085459](https://doi.org/10.1109/tcs.1984.1085459)<br><span class="ts-cite-systems">Cited by: `Chua`</span>
- Meier (2003), Presentation of Attractors with Cinema.<br><span class="ts-cite-systems">Cited by: `HyperJha`, `HyperLorenz`, `HyperYan`, `HyperYangChen`</span>
- Meleshko & Aref (1996), Phys. Fluids 8, 3215-3217. [doi:10.1063/1.869128](https://doi.org/10.1063/1.869128)<br><span class="ts-cite-systems">Cited by: `BlinkingRotlet`</span>
- Moore & Spiegel (1966), Astrophys. J. 143, 871-887. [doi:10.1086/148562](https://doi.org/10.1086/148562)<br><span class="ts-cite-systems">Cited by: `MooreSpiegel`</span>
- Nosé (1984), J. Chem. Phys. 81, 511-519; Hoover (1985), Phys. Rev. A 31, 1695-1697. [doi:10.1103/physreva.31.1695](https://doi.org/10.1103/physreva.31.1695)<br><span class="ts-cite-systems">Cited by: `NoseHoover`</span>
- Pang & Liu (2011), J. Comput. Appl. Math. 235, 2775. [doi:10.1016/j.cam.2010.11.029](https://doi.org/10.1016/j.cam.2010.11.029)<br><span class="ts-cite-systems">Cited by: `HyperPang`</span>
- Pearson (1993), Science 261, 189-192. [doi:10.1126/science.261.5118.189](https://doi.org/10.1126/science.261.5118.189)<br><span class="ts-cite-systems">Cited by: `GrayScott`</span>
- Pehlivan & Wei (2012), Turk. J. Electr. Eng. Comput. Sci. 20, 1229-1239. [doi:10.3906/elk-1103-14](https://doi.org/10.3906/elk-1103-14)<br><span class="ts-cite-systems">Cited by: `PehlivanWei`</span>
- Petrov, Scott & Showalter (1992), J. Chem. Phys. 97, 6191-6198. [doi:10.1063/1.463727](https://doi.org/10.1063/1.463727)<br><span class="ts-cite-systems">Cited by: `IsothermalChemical`</span>
- van der Pol (1926), London Edinburgh Dublin Philos. Mag. J. Sci. 2, 978-992. [doi:10.1080/14786442608564127](https://doi.org/10.1080/14786442608564127)<br><span class="ts-cite-systems">Cited by: `ForcedVanDerPol`</span>
- Prigogine (1980), From Being to Becoming, W.H. Freeman.<br><span class="ts-cite-systems">Cited by: `ForcedBrusselator`</span>
- Qi et al. (2008), Chaos Solitons Fractals 38, 705-721. [doi:10.1016/j.chaos.2006.09.012](https://doi.org/10.1016/j.chaos.2006.09.012)<br><span class="ts-cite-systems">Cited by: `QiChen`</span>
- Qi, van Wyk, van Wyk & Chen (2008), Phys. Lett. A 372, 124. [doi:10.1016/j.physleta.2007.10.082](https://doi.org/10.1016/j.physleta.2007.10.082)<br><span class="ts-cite-systems">Cited by: `HyperQi`, `Qi`</span>
- Rabinovich & Fabrikant (1979), Sov. Phys. JETP 50, 311-317.<br><span class="ts-cite-systems">Cited by: `RabinovichFabrikant`</span>
- Rikitake (1958), Proc. Cambridge Philos. Soc. 54, 89-105. [doi:10.1017/s0305004100033223](https://doi.org/10.1017/s0305004100033223)<br><span class="ts-cite-systems">Cited by: `RikitakeDynamo`</span>
- Romond, Rustici, Gonze & Goldbeter (1999), Ann. N.Y. Acad. Sci. 879, 180-193. [doi:10.1111/j.1749-6632.1999.tb10419.x](https://doi.org/10.1111/j.1749-6632.1999.tb10419.x)<br><span class="ts-cite-systems">Cited by: `CellCycle`</span>
- Rucklidge (1992), J. Fluid Mech. 237, 209-229. [doi:10.1017/s0022112092003392](https://doi.org/10.1017/s0022112092003392)<br><span class="ts-cite-systems">Cited by: `Rucklidge`</span>
- Rössler (1976), Phys. Lett. A 57, 397-398. [doi:10.1016/0375-9601(76)90101-8](https://doi.org/10.1016/0375-9601(76)90101-8)<br><span class="ts-cite-systems">Cited by: `Rossler`</span>
- Rössler (1979), Phys. Lett. A 71, 155-157. [doi:10.1016/0375-9601(79)90150-6](https://doi.org/10.1016/0375-9601(79)90150-6)<br><span class="ts-cite-systems">Cited by: `HyperRossler`</span>
- San-Um & Srisuchinwong (2012), J. Comput. 7, 1041-1047. [doi:10.4304/jcp.7.4.1041-1047](https://doi.org/10.4304/jcp.7.4.1041-1047)<br><span class="ts-cite-systems">Cited by: `SanUmSrisuchinwong`</span>
- Shadden, Lekien & Marsden (2005), Physica D 212, 271-304. [doi:10.1016/j.physd.2005.10.007](https://doi.org/10.1016/j.physd.2005.10.007)<br><span class="ts-cite-systems">Cited by: `DoubleGyre`</span>
- Shaw (1981), Z. Naturforsch. A 36, 80-112. [doi:10.1515/zna-1981-0115](https://doi.org/10.1515/zna-1981-0115)<br><span class="ts-cite-systems">Cited by: `BurkeShaw`</span>
- Shimizu & Morioka (1980), Phys. Lett. A 76, 201-204. [doi:10.1016/0375-9601(80)90466-1](https://doi.org/10.1016/0375-9601(80)90466-1)<br><span class="ts-cite-systems">Cited by: `ShimizuMorioka`</span>
- Smith, Thiffeault & Horton (2000), J. Geophys. Res. 105, 12983-12996. [doi:10.1029/1999ja000218](https://doi.org/10.1029/1999ja000218)<br><span class="ts-cite-systems">Cited by: `WindmiReduced`</span>
- Solomon & Gollub (1988), Phys. Rev. A 38, 6280-6286. [doi:10.1103/physreva.38.6280](https://doi.org/10.1103/physreva.38.6280)<br><span class="ts-cite-systems">Cited by: `OscillatingFlow`</span>
- Sprott (1994), Phys. Rev. E 50, R647-R650. [doi:10.1103/physreve.50.r647](https://doi.org/10.1103/physreve.50.r647)<br><span class="ts-cite-systems">Cited by: `SprottA`, `SprottB`, `SprottC`, `SprottD`, `SprottE`, `SprottF`, `SprottG`, `SprottH`, `SprottI`, `SprottJ`, `SprottK`, `SprottL`, `SprottM`, `SprottN`, `SprottO`, `SprottP`, `SprottQ`, `SprottR`, `SprottS`</span>
- Sprott (1997), Phys. Lett. A 228, 271-274. [doi:10.1016/s0375-9601(97)00088-1](https://doi.org/10.1016/s0375-9601(97)00088-1)<br><span class="ts-cite-systems">Cited by: `SprottJerk`</span>
- Sprott (2010), Elegant Chaos, World Scientific. [doi:10.1142/9789812838827](https://doi.org/10.1142/9789812838827)<br><span class="ts-cite-systems">Cited by: `Halvorsen`</span>
- Sprott (2011), IEEE Trans. Circuits Syst. II 58, 240-243. [doi:10.1109/tcsii.2011.2124490](https://doi.org/10.1109/tcsii.2011.2124490)<br><span class="ts-cite-systems">Cited by: `JerkCircuit`</span>
- Sprott (2014), Phys. Lett. A 378, 1361-1363. [doi:10.1016/j.physleta.2013.11.004](https://doi.org/10.1016/j.physleta.2013.11.004)<br><span class="ts-cite-systems">Cited by: `SprottTorus`</span>
- Sprott & Xiong (2015), Chaos 25, 083101. [doi:10.1063/1.4927643](https://doi.org/10.1063/1.4927643)<br><span class="ts-cite-systems">Cited by: `LorenzBounded`</span>
- Sprott (2020), Chaos Theory Appl. 2, 1-3. [doi:10.1016/j.chaos.2020.109990](https://doi.org/10.1016/j.chaos.2020.109990)<br><span class="ts-cite-systems">Cited by: `SprottMore`</span>
- Stenflo (1996), Phys. Scr. 53, 83-84. [doi:10.1088/0031-8949/53/1/015](https://doi.org/10.1088/0031-8949/53/1/015)<br><span class="ts-cite-systems">Cited by: `LorenzStenflo`</span>
- Strizhak & Kawczynski (1995), J. Phys. Chem. 99, 10830-10833. [doi:10.1021/j100027a024](https://doi.org/10.1021/j100027a024)<br><span class="ts-cite-systems">Cited by: `KawczynskiStrizhak`</span>
- Strogatz (1994), Nonlinear Dynamics and Chaos.<br><span class="ts-cite-systems">Cited by: `Torus`</span>
- Swift & Hohenberg (1977), Phys. Rev. A 15, 319-328. [doi:10.1103/physreva.15.319](https://doi.org/10.1103/physreva.15.319)<br><span class="ts-cite-systems">Cited by: `SwiftHohenberg`</span>
- Thomas (1999), Int. J. Bifurc. Chaos 9, 1889-1905. [doi:10.1142/s0218127499001383](https://doi.org/10.1142/s0218127499001383)<br><span class="ts-cite-systems">Cited by: `Thomas`</span>
- Tufillaro, Abbott & Griffiths (1984), Am. J. Phys. 52, 895-903. [doi:10.1119/1.13791](https://doi.org/10.1119/1.13791)<br><span class="ts-cite-systems">Cited by: `SwingingAtwood`</span>
- Turchin & Hanski (1997), Am. Nat. 149, 842-874. [doi:10.1086/286027](https://doi.org/10.1086/286027)<br><span class="ts-cite-systems">Cited by: `TurchinHanski`</span>
- Tuwankotta (2006), Int. J. Non-Linear Mech. 41, 180-191. [doi:10.1016/j.ijnonlinmec.2005.02.007](https://doi.org/10.1016/j.ijnonlinmec.2005.02.007)<br><span class="ts-cite-systems">Cited by: `AtmosphericRegime`</span>
- Upadhyay, Bairagi, Kundu & Chattopadhyay (2008), Appl. Math. Comput. 196, 392-401. [doi:10.1016/j.amc.2007.06.007](https://doi.org/10.1016/j.amc.2007.06.007)<br><span class="ts-cite-systems">Cited by: `SaltonSea`</span>
- Vallis (1988), J. Geophys. Res. 93, 13979-13991. [doi:10.1029/jc093ic11p13979](https://doi.org/10.1029/jc093ic11p13979)<br><span class="ts-cite-systems">Cited by: `VallisElNino`</span>
- Wang, Sun, van Wyk, Qi & van Wyk (2009), Braz. J. Phys. 39. [doi:10.1590/s0103-97332009000500007](https://doi.org/10.1590/s0103-97332009000500007)<br><span class="ts-cite-systems">Cited by: `HyperWang`, `WangSun`</span>
- Yalçın, Suykens & Vandewalle (2005), Cellular Neural Networks, Multi-Scroll Chaos and Synchronization, World Scientific. [doi:10.1142/9789812567741](https://doi.org/10.1142/9789812567741)<br><span class="ts-cite-systems">Cited by: `MultiChua`</span>
- Yanagita & Kaneko (1995), Physica D 82, 288-313. [doi:10.1016/0167-2789(94)00233-g](https://doi.org/10.1016/0167-2789(94)00233-g)<br><span class="ts-cite-systems">Cited by: `RayleighBenard`</span>
- Yu & Wang (2012), Eng. Technol. Appl. Sci. Res. 2, 209-215. [doi:10.48084/etasr.86](https://doi.org/10.48084/etasr.86)<br><span class="ts-cite-systems">Cited by: `YuWang`, `YuWang2`</span>
- Zhou & Chen (2004), Int. J. Bifurcation Chaos. [doi:10.1142/s0218127404010175](https://doi.org/10.1142/s0218127404010175)<br><span class="ts-cite-systems">Cited by: `ZhouChen`</span>

### Delay differential equations

- Driver (1977), Ordinary and Delay Differential Equations, Springer. [doi:10.1007/978-1-4684-9467-9_5](https://doi.org/10.1007/978-1-4684-9467-9_5)<br><span class="ts-cite-systems">Cited by: `ScrollDelay`</span>
- Ikeda & Matsumoto (1987), Physica D 29, 223-235. [doi:10.1016/0167-2789(87)90058-3](https://doi.org/10.1016/0167-2789(87)90058-3)<br><span class="ts-cite-systems">Cited by: `IkedaDelay`</span>
- Mackey & Glass (1977), Science 197, 287-289. [doi:10.1126/science.267326](https://doi.org/10.1126/science.267326)<br><span class="ts-cite-systems">Cited by: `MackeyGlass`</span>
- Sprott (2007), Physics Letters A 366, 397-402. [doi:10.1016/j.physleta.2007.01.083](https://doi.org/10.1016/j.physleta.2007.01.083)<br><span class="ts-cite-systems">Cited by: `SprottDelay`</span>
- Tamasevicius, Mykolaitis & Bumeliene (2006), Electron. Lett. 42, 13. [doi:10.1049/el:20061245](https://doi.org/10.1049/el:20061245)<br><span class="ts-cite-systems">Cited by: `PiecewiseCircuit`</span>

### Stochastic differential equations

- Kramers (1940), Physica 7, 284-304. [doi:10.1016/S0031-8914(40)90098-2](https://doi.org/10.1016/S0031-8914(40)90098-2)<br><span class="ts-cite-systems">Cited by: `DoubleWell`</span>
- Osborne (1959), Oper. Res. 7, 145-173. [doi:10.1287/opre.7.2.145](https://doi.org/10.1287/opre.7.2.145)<br><span class="ts-cite-systems">Cited by: `GeometricBrownianMotion`</span>
- Uhlenbeck & Ornstein (1930), Phys. Rev. 36, 823-841. [doi:10.1103/PhysRev.36.823](https://doi.org/10.1103/PhysRev.36.823)<br><span class="ts-cite-systems">Cited by: `OrnsteinUhlenbeck`</span>

### Discrete maps

- Adler & Rivlin (1964), Proc. Amer. Math. Soc. 15, 794-796. [doi:10.1090/s0002-9939-1964-0202968-3](https://doi.org/10.1090/s0002-9939-1964-0202968-3)<br><span class="ts-cite-systems">Cited by: `Chebyshev`</span>
- Arnold (1965), Amer. Math. Soc. Transl. 46, 213-284.<br><span class="ts-cite-systems">Cited by: `Circle`</span>
- Baier & Klein (1990), Phys. Lett. A 151, 281-284. [doi:10.1016/0375-9601(90)90283-t](https://doi.org/10.1016/0375-9601(90)90283-t)<br><span class="ts-cite-systems">Cited by: `GeneralizedHenon`</span>
- Bogdanov (1981), Selecta Math. Soviet. 1, 389-421.<br><span class="ts-cite-systems">Cited by: `Bogdanov`</span>
- Chirikov (1979), Phys. Rep. 52, 263-379. [doi:10.1016/0370-1573(79)90023-1](https://doi.org/10.1016/0370-1573(79)90023-1)<br><span class="ts-cite-systems">Cited by: `Chirikov`</span>
- Devaney (1984), Physica D 10, 387-393. [doi:10.1016/0167-2789(84)90187-8](https://doi.org/10.1016/0167-2789(84)90187-8)<br><span class="ts-cite-systems">Cited by: `Gingerbreadman`</span>
- Dewdney (1986), Scientific American 255(3), 14-20. [doi:10.1038/scientificamerican0986-14](https://doi.org/10.1038/scientificamerican0986-14)<br><span class="ts-cite-systems">Cited by: `Hopalong`</span>
- Dewdney (1987), Scientific American 257(1), 108-111. [doi:10.1038/scientificamerican0787-108](https://doi.org/10.1038/scientificamerican0787-108)<br><span class="ts-cite-systems">Cited by: `DeJong`</span>
- Gumowski & Mira (1980), Recurrences and Discrete Dynamic Systems. [doi:10.1007/bfb0089135](https://doi.org/10.1007/bfb0089135)<br><span class="ts-cite-systems">Cited by: `GumowskiMira`</span>
- Hilborn (2000), Chaos and Nonlinear Dynamics, 2nd ed. (Oxford University Press). [doi:10.1093/acprof:oso/9780198507239.001.0001](https://doi.org/10.1093/acprof:oso/9780198507239.001.0001)<br><span class="ts-cite-systems">Cited by: `Gauss`</span>
- Hopf (1937), Ergodentheorie (Springer, Berlin). [doi:10.1007/978-3-642-86630-2](https://doi.org/10.1007/978-3-642-86630-2)<br><span class="ts-cite-systems">Cited by: `Baker`</span>
- Hénon (1976), Commun. Math. Phys. 50, 69-77. [doi:10.1007/bf01608556](https://doi.org/10.1007/bf01608556)<br><span class="ts-cite-systems">Cited by: `Henon`</span>
- Ikeda (1979), Opt. Commun. 30, 257-261. [doi:10.1016/0030-4018(79)90090-7](https://doi.org/10.1016/0030-4018(79)90090-7)<br><span class="ts-cite-systems">Cited by: `Ikeda`</span>
- Kaplan & Yorke (1979), Functional Differential Equations and Approximation of Fixed Points, Lecture Notes in Mathematics 730, 204-227. [doi:10.1007/bfb0064319](https://doi.org/10.1007/bfb0064319)<br><span class="ts-cite-systems">Cited by: `KaplanYorke`</span>
- May (1976), Nature 261, 459-467. [doi:10.1038/261459a0](https://doi.org/10.1038/261459a0)<br><span class="ts-cite-systems">Cited by: `Logistic`</span>
- Nusse & Yorke (1994), Dynamics: Numerical Explorations. [doi:10.1007/978-1-4684-0231-5](https://doi.org/10.1007/978-1-4684-0231-5)<br><span class="ts-cite-systems">Cited by: `Tinkerbell`</span>
- Pickover (1990), Computers, Pattern, Chaos and Beauty (St. Martin's Press).<br><span class="ts-cite-systems">Cited by: `Pickover`</span>
- Ricker (1954), J. Fish. Res. Board Can. 11, 559-623. [doi:10.1139/f54-039](https://doi.org/10.1139/f54-039)<br><span class="ts-cite-systems">Cited by: `Ricker`</span>
- Rössler (1979), 'Chaotic oscillations: an example of hyperchaos', Lectures in Applied Mathematics 17, 141-156.<br><span class="ts-cite-systems">Cited by: `FoldedTowel`</span>
- Maynard Smith (1968), Mathematical Ideas in Biology (Cambridge University Press). [doi:10.1017/cbo9780511565144](https://doi.org/10.1017/cbo9780511565144)<br><span class="ts-cite-systems">Cited by: `MaynardSmith`</span>
- Classical map; see e.g. Strogatz, Nonlinear Dynamics and Chaos.<br><span class="ts-cite-systems">Cited by: `Tent`</span>
- Ulam & von Neumann (1947), Bull. Amer. Math. Soc. 53, 1120.<br><span class="ts-cite-systems">Cited by: `Ulam`</span>
- Zaslavsky (1978), Phys. Lett. A 69, 145-147. [doi:10.1016/0375-9601(78)90195-0](https://doi.org/10.1016/0375-9601(78)90195-0)<br><span class="ts-cite-systems">Cited by: `Zaslavskii`</span>
- Zeraoulia & Sprott (2011), Int. J. Bifurcation Chaos 21, 155-160. [doi:10.1142/s0218127411028325](https://doi.org/10.1142/s0218127411028325)<br><span class="ts-cite-systems">Cited by: `ZeraouliaSprott`</span>

## Methods {#methods}

The primary literature behind the analysis toolkit. These are the citations that close each page in the [Analysis](../analysis/index.md) section, collected here by topic; every implementation names its source in its docstring and on its prose page.

### Lyapunov exponents & tangent-space dynamics

- G. Benettin, L. Galgani & J.-M. Strelcyn, "Kolmogorov entropy and numerical experiments", *Phys. Rev. A* **14**, 2338 (1976). [doi:10.1103/PhysRevA.14.2338](https://doi.org/10.1103/PhysRevA.14.2338)
- G. Benettin, L. Galgani, A. Giorgilli & J.-M. Strelcyn, “Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them”, *Meccanica* **15**, 9 & 21 (1980). [doi:10.1007/BF02128236](https://doi.org/10.1007/BF02128236)
- H. Kantz, "A robust method to estimate the maximal Lyapunov exponent of a time series", *Phys. Lett. A* **185**, 77 (1994). [doi:10.1016/0375-9601(94)90991-1](https://doi.org/10.1016/0375-9601(94)90991-1)
- J. L. Kaplan & J. A. Yorke, “Chaotic behavior of multidimensional difference equations”, in *Functional Differential Equations and Approximation of Fixed Points*, Lecture Notes in Mathematics **730**, 204, Springer (1979). [doi:10.1007/BFb0064319](https://doi.org/10.1007/BFb0064319)
- M. T. Rosenstein, J. J. Collins & C. J. De Luca, “A practical method for calculating largest Lyapunov exponents from small data sets”, *Physica D* **65**, 117 (1993). [doi:10.1016/0167-2789(93)90009-P](https://doi.org/10.1016/0167-2789(93)90009-P)

### Chaos indicators

- G. A. Gottwald & I. Melbourne, “A new test for chaos in deterministic systems”, *Proc. R. Soc. Lond. A* **460**, 603 (2004). [doi:10.1098/rspa.2003.1183](https://doi.org/10.1098/rspa.2003.1183)
- G. A. Gottwald & I. Melbourne, “On the implementation of the 0–1 test for chaos”, *SIAM J. Appl. Dyn. Syst.* **8**, 129 (2009). [doi:10.1137/080718851](https://doi.org/10.1137/080718851)
- B. R. Hunt & E. Ott, "Defining chaos", *Chaos* **25**, 097618 (2015). [doi:10.1063/1.4922973](https://doi.org/10.1063/1.4922973)
- Ch. Skokos, T. C. Bountis & Ch. Antonopoulos, “Geometrical properties of local dynamics in Hamiltonian systems: the Generalized Alignment Index (GALI) method”, *Physica D* **231**, 30 (2007). [doi:10.1016/j.physd.2007.04.004](https://doi.org/10.1016/j.physd.2007.04.004)

### Fractal dimensions

- R. Badii & A. Politi, “Statistical description of chaotic attractors: the dimension function”, *J. Stat. Phys.* **40**, 725 (1985). [doi:10.1007/BF01009897](https://doi.org/10.1007/BF01009897)
- P. Grassberger & I. Procaccia, “Characterization of strange attractors”, *Phys. Rev. Lett.* **50**, 346 (1983). [doi:10.1103/PhysRevLett.50.346](https://doi.org/10.1103/PhysRevLett.50.346)
- P. Grassberger, “Generalizations of the Hausdorff dimension of fractal measures”, *Phys. Lett. A* **107**, 101 (1985). [doi:10.1016/0375-9601(85)90724-8](https://doi.org/10.1016/0375-9601(85)90724-8)
- H. G. E. Hentschel & I. Procaccia, “The infinite number of generalized dimensions of fractals and strange attractors”, *Physica D* **8**, 435 (1983). [doi:10.1016/0167-2789(83)90235-X](https://doi.org/10.1016/0167-2789(83)90235-X)
- J. Theiler, "Estimating fractal dimension", *J. Opt. Soc. Am. A* **7**, 1055 (1990). [doi:10.1364/JOSAA.7.001055](https://doi.org/10.1364/JOSAA.7.001055)

### Delay embedding & state-space reconstruction

- L. Cao, “Practical method for determining the minimum embedding dimension of a scalar time series”, *Physica D* **110**, 43 (1997). [doi:10.1016/S0167-2789(97)00118-8](https://doi.org/10.1016/S0167-2789(97)00118-8)
- A. M. Fraser & H. L. Swinney, “Independent coordinates for strange attractors from mutual information”, *Phys. Rev. A* **33**, 1134 (1986). [doi:10.1103/PhysRevA.33.1134](https://doi.org/10.1103/PhysRevA.33.1134)
- M. B. Kennel, R. Brown & H. D. I. Abarbanel, “Determining embedding dimension for phase-space reconstruction using a geometrical construction”, *Phys. Rev. A* **45**, 3403 (1992). [doi:10.1103/PhysRevA.45.3403](https://doi.org/10.1103/PhysRevA.45.3403)
- F. Takens, "Detecting strange attractors in turbulence", in *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics **898**, 366, Springer (1981). [doi:10.1007/BFb0091924](https://doi.org/10.1007/BFb0091924)

### Entropy & complexity

- C. Bandt & B. Pompe, “Permutation entropy: a natural complexity measure for time series”, *Phys. Rev. Lett.* **88**, 174102 (2002). [doi:10.1103/PhysRevLett.88.174102](https://doi.org/10.1103/PhysRevLett.88.174102)
- M. Costa, A. L. Goldberger & C.-K. Peng, “Multiscale entropy analysis of complex physiologic time series”, *Phys. Rev. Lett.* **89**, 068102 (2002). [doi:10.1103/PhysRevLett.89.068102](https://doi.org/10.1103/PhysRevLett.89.068102)
- B. Fadlallah, B. Chen, A. Keil & J. Príncipe, “Weighted-permutation entropy: a complexity measure for time series incorporating amplitude information”, *Phys. Rev. E* **87**, 022911 (2013). [doi:10.1103/PhysRevE.87.022911](https://doi.org/10.1103/PhysRevE.87.022911)
- F. Kaspar & H. G. Schuster, “Easily calculable measure for the complexity of spatiotemporal patterns”, *Phys. Rev. A* **36**, 842 (1987). [doi:10.1103/PhysRevA.36.842](https://doi.org/10.1103/PhysRevA.36.842)
- A. Lempel & J. Ziv, "On the complexity of finite sequences", *IEEE Trans. Inf. Theory* **22**, 75 (1976). [doi:10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501)
- S. M. Pincus, “Approximate entropy as a measure of system complexity”, *Proc. Natl. Acad. Sci. USA* **88**, 2297 (1991). [doi:10.1073/pnas.88.6.2297](https://doi.org/10.1073/pnas.88.6.2297)
- J. S. Richman & J. R. Moorman, “Physiological time-series analysis using approximate entropy and sample entropy”, *Am. J. Physiol. Heart Circ. Physiol.* **278**, H2039 (2000). [doi:10.1152/ajpheart.2000.278.6.H2039](https://doi.org/10.1152/ajpheart.2000.278.6.H2039)
- M. Rostaghi & H. Azami, “Dispersion entropy: a measure for time-series analysis”, *IEEE Signal Process. Lett.* **23**, 610 (2016). [doi:10.1109/LSP.2016.2542881](https://doi.org/10.1109/LSP.2016.2542881)

### Recurrence & RQA

- J.-P. Eckmann, S. O. Kamphorst & D. Ruelle, “Recurrence plots of dynamical systems”, *Europhys. Lett.* **4**, 973 (1987). [doi:10.1209/0295-5075/4/9/004](https://doi.org/10.1209/0295-5075/4/9/004)
- N. Marwan, M. C. Romano, M. Thiel & J. Kurths, “Recurrence plots for the analysis of complex systems”, *Phys. Rep.* **438**, 237 (2007). [doi:10.1016/j.physrep.2006.11.001](https://doi.org/10.1016/j.physrep.2006.11.001)
- L. L. Trulla, A. Giuliani, J. P. Zbilut & C. L. Webber, “Recurrence quantification analysis of the logistic equation with transients”, *Phys. Lett. A* **223**, 255 (1996). [doi:10.1016/S0375-9601(96)00741-4](https://doi.org/10.1016/S0375-9601(96)00741-4)
- J. P. Zbilut & C. L. Webber, “Embeddings and delays as derived from quantification of recurrence plots”, *Phys. Lett. A* **171**, 199 (1992). [doi:10.1016/0375-9601(92)90426-M](https://doi.org/10.1016/0375-9601(92)90426-M)

### Surrogates & nonlinearity tests

- C. Diks, J. C. van Houwelingen, F. Takens & J. DeGoede, “Reversibility as a criterion for discriminating time series”, *Phys. Lett. A* **201**, 221 (1995). [doi:10.1016/0375-9601(95)00239-Y](https://doi.org/10.1016/0375-9601(95)00239-Y)
- H. Kantz & T. Schreiber, *Nonlinear Time Series Analysis*, 2nd ed., Cambridge University Press (2004). [doi:10.1017/CBO9780511755798](https://doi.org/10.1017/CBO9780511755798)
- T. Schreiber & A. Schmitz, “Improved surrogate data for nonlinearity tests”, *Phys. Rev. Lett.* **77**, 635 (1996). [doi:10.1103/PhysRevLett.77.635](https://doi.org/10.1103/PhysRevLett.77.635)
- G. Sugihara & R. M. May, “Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series”, *Nature* **344**, 734 (1990). [doi:10.1038/344734a0](https://doi.org/10.1038/344734a0)
- J. Theiler, "Spurious dimension from correlation algorithms applied to limited time-series data", *Phys. Rev. A* **34**, 2427 (1986). [doi:10.1103/PhysRevA.34.2427](https://doi.org/10.1103/PhysRevA.34.2427)
- J. Theiler, S. Eubank, A. Longtin, B. Galdrikian & J. D. Farmer, “Testing for nonlinearity in time series: the method of surrogate data”, *Physica D* **58**, 77 (1992). [doi:10.1016/0167-2789(92)90102-S](https://doi.org/10.1016/0167-2789(92)90102-S)

### Fixed points, periodic orbits & interval methods

- R. L. Davidchack & Y.-C. Lai, “Efficient algorithm for detecting unstable periodic orbits in chaotic systems”, *Phys. Rev. E* **60**, 6172 (1999). [doi:10.1103/PhysRevE.60.6172](https://doi.org/10.1103/PhysRevE.60.6172)
- R. Krawczyk, “Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken”, *Computing* **4**, 187 (1969). [doi:10.1007/BF02234767](https://doi.org/10.1007/BF02234767)
- A. Neumaier, *Interval Methods for Systems of Equations*, Cambridge University Press (1990).
- P. Schmelcher & F. K. Diakonos, “Detecting unstable periodic orbits of chaotic dynamical systems”, *Phys. Rev. Lett.* **78**, 4733 (1997). [doi:10.1103/PhysRevLett.78.4733](https://doi.org/10.1103/PhysRevLett.78.4733)

### Attractors, basins & global stability

- G. Datseris & A. Wagemakers, “Effortless estimation of basins of attraction”, *Chaos* **32**, 023104 (2022). [doi:10.1063/5.0076568](https://doi.org/10.1063/5.0076568)
- G. Datseris, K. L. Rossi & A. Wagemakers, “Framework for global stability analysis of dynamical systems”, *Chaos* **33**, 073151 (2023). [doi:10.1063/5.0159675](https://doi.org/10.1063/5.0159675)
- A. Daza, A. Wagemakers, M. A. F. Sanjuán & J. A. Yorke, “Testing for basins of Wada”, *Sci. Rep.* **5**, 16579 (2015). [doi:10.1038/srep16579](https://doi.org/10.1038/srep16579)
- A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin & M. A. F. Sanjuán, “Basin entropy: a new tool to analyze uncertainty in dynamical systems”, *Sci. Rep.* **6**, 31416 (2016). [doi:10.1038/srep31416](https://doi.org/10.1038/srep31416)
- C. Grebogi, S. W. McDonald, E. Ott & J. A. Yorke, “Final state sensitivity: an obstruction to predictability”, *Phys. Lett. A* **99**, 415 (1983). [doi:10.1016/0375-9601(83)90945-3](https://doi.org/10.1016/0375-9601(83)90945-3)
- L. Halekotte & U. Feudel, “Minimal fatal shocks in multistable complex networks”, *Sci. Rep.* **10**, 11783 (2020). [doi:10.1038/s41598-020-68805-6](https://doi.org/10.1038/s41598-020-68805-6)
- P. J. Menck, J. Heitzig, N. Marwan & J. Kurths, “How basin stability complements the linear-stability paradigm”, *Nat. Phys.* **9**, 89 (2013). [doi:10.1038/nphys2516](https://doi.org/10.1038/nphys2516)

### Orbit diagrams, Poincaré maps & the classics

- M. J. Feigenbaum, “Quantitative universality for a class of nonlinear transformations”, *J. Stat. Phys.* **19**, 25 (1978). [doi:10.1007/BF01020332](https://doi.org/10.1007/BF01020332)
- M. Hénon, “On the numerical computation of Poincaré maps”, *Physica D* **5**, 412 (1982). [doi:10.1016/0167-2789(82)90034-3](https://doi.org/10.1016/0167-2789(82)90034-3)
- R. M. May, “Simple mathematical models with very complicated dynamics”, *Nature* **261**, 459 (1976). [doi:10.1038/261459a0](https://doi.org/10.1038/261459a0)
- H. Poincaré, *Les méthodes nouvelles de la mécanique céleste*, Gauthier-Villars (1892–1899).

---

*The systems bibliography is generated from the registry by `docs/_tooling/make_bibliography.py`; re-run it after adding or editing a system. If a DOI resolves to the wrong paper, fix the `doi` class attribute on the system, not this page.*
