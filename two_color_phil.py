from iotbx.phil import parse

two_color_phil_scope = parse('''
  indexing {
    two_color {
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
      metal_foil_correction
        .help = Use if the detector is partially obscured by a metal foil designed to \
                absorb one of the energies completely and the other partially
      {
        absorption_edge_energy = None
          .type = float
          .help = Reflections whose energy is higher than this energy are discarded. \
                  Reflections whose energy is lower than this energy are attenuated
        transmittance = None
          .type = float
          .help = Fractional transmittance for the partially absorbed energy, assuming \
                  normal incidence with respect to the detector
        two_theta_deg = None
          .type = float
          .help = Reflections with two theta angles less than this value (in degrees) \
                  will be corrected.
      }
    }
  }
  calc_G_and_B {
    do_calc = False
      .type = bool
      .help = Can choose to postrefine G and B factors for a still
    include scope xfel.command_line.cxi_merge.master_phil
  }
''', process_includes=True)


