from iotbx.phil import parse
from dials.command_line.stills_process import phil_scope

two_color_phil_scope = parse('''
  indexing {
    two_color {
      optimize_initial_basis_vectors = False
        .type = bool
        .help = whether to optimize basis vectors after grid search 
      debug = False
        .type = bool
        .help = Reflections for both wavelengths at all hkls
      low_energy = 1.0
        .type = float
        .help = Low energy value in eV
      high_energy = 2.0
        .type = float
        .help = High energy value in eV
      avg_energy = 3.0
        .type = float
        .help = The average of both energies (eV) used for overlapped spots
      filter_by_mag = None
        .type = floats
        .help = "a tuple , first element specifies the percentage deviation"
                "for the unit cell lengths allowed in the candidate crystal"
                "the second element is absolute angle deviation allowed"
      spiral_method = None
        .type = floats
        .help = "a tuple, first element is the scale of the noise"
                "applied to the grid search gaussian jitter, spe-"
                "cifically basis vectors are sampled about there mean" 
                "with this this number being the variance. The second element"
                "is the number of points to sample on the hemisphere, I was"
                "usng ~800,000 oer unit cell dimension, each receives its own"
                "gaussian jitter."  
      spiral_seed = None
        .type = int
        .help = "an integer to seed the random number generator"
      n_unique_v = 30
        .type = int
        .help = "number of unique basis vectors chosen from the grid"
                "search based after sorting by functional value"
      block_size = 50
        .type = int
        .help = "How many blocksto  divide the RL vectors up in the block size comp"
                "during the grid search"
      }
    }
''', process_includes=True)

phil_scope.adopt_scope(two_color_phil_scope)

params = phil_scope.extract()
# these parameters are important for two color indexer
params.indexing.refinement_protocol.mode = None
params.indexing.stills.refine_all_candidates = False
