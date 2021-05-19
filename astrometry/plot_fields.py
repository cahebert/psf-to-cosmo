"""

"""
import os
import glob
import matplotlib.pyplot as plt
import seaborn
seaborn.set_palette('bright')

import astrometry_field

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    """
    Main function for testing and debugging purposes.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',type=str,required=False,default='output',
                        help='Input directory of data files [dir]')
    parser.add_argument('--outdir',type=str,required=False,default='plots',
                        help='Output directory of plots [dir]')
    args = parser.parse_args()

    #---------------------------------------------------------------------------
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    #---------------------------------------------------------------------------

    infiles = glob.glob(f'{args.indir}/*.pkl')

    for infile in infiles:
        print(f'Making plots for {os.path.basename(infile)}...')
        af = astrometry_field.AstrometryField(infile)
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        af.plot_astrometric_residuals(axs[0], af.xs, af.ys)
        af.plot_2pcf(axs[1], af.dr, af.xi0, af.xi1)

        plt.suptitle(f'{os.path.basename(infile)}')
        plt.tight_layout()
        plt.savefig(f'{args.outdir}/{os.path.splitext(os.path.basename(infile))[0]}.png', bbox_inches='tight')
        plt.close()
