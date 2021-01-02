"""
formation evaluation
Author: Olaide Joseph
Email: Josepholaide10@gmail.com
"""

import lasio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import warnings
plt.style.use('ggplot')
warnings.filterwarnings("ignore")


class formation_eval:
    """
        Evaluates the formation and determines formation characteristic such as shale volume,
        reservoir and non-reservoir zones.
        args::
            datapath: LAS datapath
            mnemonics: list of well log mnemonics, if 'None', density, neutron, Gamma ray, SP and resistivity
            logs are passed if available.
        """

    def __init__(self, datapath: str = None, mnemonics: list = None):
        """

        :type datapath: str
        :type mnemonics: list

        """
        if mnemonics is None:
            self.mnemonics = mnemonics
        if datapath is None:
            pass
        else:
            self.datapath = datapath

    def read_lasio(self):
        """
        reads the .LAS file and converts to a dataframe
        :return: log data description and log dataframe
        """
        path = self.datapath
        lasfile = lasio.read(path)
        df = lasfile.df()
        if 'D' or 'E' or 'P' in df.index.name:
            df.reset_index(inplace=True)
        return lasfile.header, df

    def well_logs(self, dataframe: pd.DataFrame):
        """
        filters the dataframe and returns a dataframe with the specified mnemonics or necessary and available mnemonics
        :param dataframe: pandas dataframe object
        :return: well log dataframe
        """
        if self.mnemonics is None:
            logs = ['DEPTH', 'ROP', 'GR', 'SP', 'CALI', 'BS', 'RD', 'RT', 'RM', 'RS', 'NPHI',
                    'CNL', 'RHOB', 'DRHO', 'PHID', 'DT', 'PEF']
            logsnew = []
            for i in logs:
                if i in dataframe.columns:
                    logsnew.append(i)
        else:
            logs = self.mnemonics
            logsnew = []
            for i in logs:
                if i in dataframe.columns:
                    logsnew.append(i)
        well_logs = logsnew
        if 'GR' in well_logs:
            dataframe['GR'] = np.where(dataframe['GR'] > 150, 150, dataframe['GR'])
            dataframe['GR'] = np.where(dataframe['GR'] < 0, 0, dataframe['GR'])
        if 'NPHI' in well_logs:
            dataframe['NPHI'] = np.where(dataframe['NPHI'] > 0.50, 0.50, dataframe['NPHI'])
            dataframe['NPHI'] = np.where(dataframe['NPHI'] < -0.15, -0.15, dataframe['NPHI'])
        return dataframe[well_logs]

    @staticmethod
    def lognan_cleaning(dataframe: pd.DataFrame, fill_value: int or float = None):
        """
        This fills the missing numbers in the dataframe
        :param fill_value: value to replace missing number with, if 'None', mean is used.
        :param dataframe: pandas dataframe
        :rtype: pd.Dataframe
        """
        df = dataframe.copy()
        if fill_value is None:
            df.fillna(np.mean(df), inplace=True)
        else:
            df.fillna(fill_value, axis=1, inplace=True)

        return df

    @staticmethod
    def log_viz(data, min_depth: float or int or None = None, max_depth: float or int or None = None,
                plotsize: tuple = None):
        """
        well log plots
        :param data: well logs dataframe
        :param min_depth: top of reservoir, closer to the surface (length units)
        :param max_depth: bottom of reservoir, closer to the subsurface (length units)
        :param plotsize: the plot figsize in tuple form
        """
        if plotsize is None:
            plotsize = (18, 15)
        logs = data.columns
        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=plotsize, sharey=False)
        fig.suptitle("Logs Visualization", fontsize=22, y=1.02)

        # General setting for all axis
        for axes in ax:
            axes.get_xaxis().set_visible(False)
            axes.invert_yaxis()
            axes.spines['left'].set_color('k')
            axes.spines['right'].set_color('k')
            axes.minorticks_on()

            if min_depth and max_depth is not None:
                axes.set_ylim(max_depth, min_depth)
            else:
                axes.set_ylim(data['DEPTH'].max(), data['DEPTH'].min())

        # 1st track: CALI, BS
        if 'CALI' and 'BS' not in logs:
            fig.delaxes(ax=ax[0])
            ax[1].set_ylim(max_depth, min_depth)
        else:
            ax[0].minorticks_on()
            ax[0].grid(b=True, which='major', color='black', linestyle='--')
            ax[0].grid(b=True, which='minor', color='grey', linestyle=':')

            if 'CALI' in logs:
                cali = ax[0].twiny()
                cali.minorticks_on()
                cali.set_xlim(6, 26)
                cali.plot(data.CALI, data.DEPTH, label='CALI[in]', color='red')
                cali.spines['top'].set_position(('outward', 20))
                cali.spines['top'].set_color('r')
                cali.set_xlabel('CALI[in]', color='red')
                cali.tick_params(axis='x', colors='red')
                cali.grid(b=True, which='major', color='k', linestyle='--')
                cali.grid(b=True, which='minor', color='grey', linestyle=':')
            else:
                pass

            if 'BS' in logs:
                bs = ax[0].twiny()
                bs.minorticks_on()
                bs.set_xlim(6, 26)
                bs.plot(data.BS, data.DEPTH, label='BS[in]', color='y')
                bs.spines['top'].set_position(('outward', 60))
                bs.spines['top'].set_color('y')
                bs.set_xlabel('BS[in]', color='y')
                bs.tick_params(axis='x', colors='y')
                bs.grid(b=True, which='major', color='k', linestyle='--')
                bs.grid(b=True, which='minor', color='grey', linestyle=':')
            else:
                pass

        # 2nd track: GR, SP
        if 'CALI' and 'BS' not in logs:
            ax[1].set_ylim(max_depth, min_depth)
        ax[1].minorticks_on()
        ax[1].grid(b=True, which='major', color='black', linestyle='--')
        ax[1].grid(b=True, which='minor', color='grey', linestyle=':')

        if 'GR' in logs:
            gr = ax[1].twiny()
            gr.minorticks_on()
            gr.set_xlim(0, 150)
            gr.plot(data.GR, data.DEPTH, label='GR[api]', color='green')
            gr.spines['top'].set_position(('outward', 20))
            gr.spines['top'].set_color('g')
            gr.set_xlabel('GR[api]', color='green')
            gr.tick_params(axis='x', colors='green')

            gr.grid(b=True, which='major', color='k', linestyle='--')
            gr.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'SP' in logs:
            sp = ax[1].twiny()
            sp.minorticks_on()
            sp.set_xlim(data['SP'].min(), data.max())
            sp.plot(data.GR, data.DEPTH, label='SP[mV]', color='b')
            sp.spines['top'].set_position(('outward', 60))
            sp.spines['top'].set_color('b')
            sp.set_xlabel('GR[api]', color='b')
            sp.tick_params(axis='x', colors='b')

            sp.grid(b=True, which='major', color='k', linestyle='--')
            sp.grid(b=True, which='minor', color='grey', linestyle=':')

        # 3rd track: resistivity track
        ax[2].grid(b=True, which='major', color='black', linestyle='--')
        ax[2].grid(b=True, which='minor', color='grey', linestyle=':')
        ax[2].minorticks_on()

        if 'RD' in logs:
            rd = ax[2].twiny()
            rd.minorticks_on()
            rd.set_xlim(0.2, 2500)
            rd.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rd.spines['top'].set_position(('outward', 10))
            rd.spines['top'].set_color('k')
            rd.semilogx(data.RD, data.DEPTH, '--', linewidth=1, c='black')
            rd.set_xlabel('RD [ohm.m]', color='black')
            rd.tick_params(axis='x', colors='black')
            rd.grid(b=True, which='major', color='black', linestyle='--')
            rd.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RS' in logs:
            rs = ax[2].twiny()
            rs.minorticks_on()
            rs.set_xlim(0.2, 2500)
            rs.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rs.spines['top'].set_position(('outward', 50))
            rs.spines['top'].set_color('b')
            rs.semilogx(data.RS, data.DEPTH, linewidth=1, c='b', )
            rs.set_xlabel('RS [ohm.m]', color='b')
            rs.tick_params(axis='x', colors='b')
            rs.grid(b=True, which='major', color='black', linestyle='--')
            rs.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RM' in logs:
            rm = ax[2].twiny()
            rm.minorticks_on()
            rm.set_xlim(0.2, 2500)
            rm.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rm.spines['top'].set_position(('outward', 90))
            rm.spines['top'].set_color('gold')
            rm.semilogx(data.RM, data.DEPTH, linewidth=1, c='gold', )
            rm.set_xlabel('RM [ohm.m]', color='gold')
            rm.tick_params(axis='x', colors='gold')
            rm.grid(b=True, which='major', color='black', linestyle='--')
            rm.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RT' in logs:
            rt = ax[2].twiny()
            rt.minorticks_on()
            rt.set_xlim(0.2, 2500)
            rt.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rt.spines['top'].set_position(('outward', 130))
            rt.spines['top'].set_color('brown')
            rt.semilogx(data.RT, data.DEPTH, linewidth=1, c='brown', )
            rt.set_xlabel('RT [ohm.m]', color='brown')
            rt.tick_params(axis='x', colors='brown')
            rt.grid(b=True, which='major', color='black', linestyle='--')
            rt.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RXO' in logs:
            rxo = ax[2].twiny()
            rxo.minorticks_on()
            rxo.set_xlim(0.2, 2500)
            rxo.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rxo.spines['top'].set_position(('outward', 170))
            rxo.spines['top'].set_color('c')
            rxo.semilogx(data.RXO, data.DEPTH, linewidth=1, c='c', )
            rxo.set_xlabel('RXO [ohm.m]', color='c')
            rxo.tick_params(axis='x', colors='c')
            rxo.grid(b=True, which='major', color='black', linestyle='--')
            rxo.grid(b=True, which='minor', color='grey', linestyle=':')

        # 4th track NPHI, DPHI, RHOB
        ax[3].minorticks_on()
        ax[3].grid(b=True, which='major', color='black', linestyle='--')
        ax[3].grid(b=True, which='minor', color='grey', linestyle=':')

        if 'NPHI' in logs:
            nphi = ax[3].twiny()
            nphi.minorticks_on()
            nphi.set_xlim(0.45, -0.15)
            nphi.spines['top'].set_position(('outward', 20))
            nphi.spines['top'].set_color('blue')
            nphi.set_xlabel("v/v")
            nphi.plot(data.NPHI, data.DEPTH, linewidth=1, label='v/v', color='blue')
            nphi.set_xlabel('NPHI [v/v]', color='blue')
            nphi.tick_params(axis='x', colors='blue')
            nphi.xaxis.set_major_locator(plt.MultipleLocator(0.2))

            nphi.grid(b=True, which='major', color='black', linestyle='--')
            nphi.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RHOB' in logs:
            rhob = ax[3].twiny()
            rhob.set_xlim(1.95, 2.95)
            rhob.plot(data.RHOB, data.DEPTH, '--', linewidth=1, label='g/cm^3', color='red')
            rhob.spines['top'].set_position(('outward', 60))
            rhob.spines['top'].set_color('red')
            rhob.set_xlabel('RHOB [g/cm^3]', color='red')
            rhob.tick_params(axis='x', colors='red')
            rhob.xaxis.set_major_locator(plt.MultipleLocator(0.4))

        elif 'PHID' in logs:
            phid = ax[3].twiny()
            phid.set_xlim(0.45, -0.15)
            phid.plot(data.PHID, data.DEPTH, '--', linewidth=1, label='%', color='red')
            phid.spines['top'].set_position(('outward', 60))
            phid.spines['top'].set_color('red')
            phid.set_xlabel('PHID [%]', color='red')
            phid.tick_params(axis='x', colors='red')
            phid.xaxis.set_major_locator(plt.MultipleLocator(0.4))

        if 'NPHI' and 'RHOB' in logs:
            # https://stackoverflow.com/questions/57766457/how-to-plot-fill-betweenx-to-fill-the-area-between-y1-and-y2-with-different-scal
            x2p, _ = (rhob.transData + nphi.transData.inverted()).transform(np.c_[data.RHOB, data.DEPTH]).T
            nphi.autoscale(False)
            nphi.fill_betweenx(data.DEPTH, data.NPHI, x2p, color="goldenrod", alpha=0.4, where=(x2p > data.NPHI))
            nphi.fill_betweenx(data.DEPTH, data.NPHI, x2p, color="turquoise", alpha=0.4, where=(x2p < data.NPHI))

        # 5th DT and PEF
        if 'PEF' and 'DT' not in logs:
            fig.delaxes(ax=ax[4])
        else:
            ax[4].minorticks_on()
            ax[4].grid(b=True, which='major', color='black', linestyle='--')
            ax[4].grid(b=True, which='minor', color='grey', linestyle=':')

            if 'DT' in logs:
                dt = ax[4].twiny()
                dt.minorticks_on()
                dt.set_xlim(200, 40)
                dt.spines['top'].set_position(('outward', 20))
                dt.spines['top'].set_color('c')
                dt.plot(data.DT, data.DEPTH, linewidth=1, label="US/F", color='c')
                dt.set_xlabel("DT", color='c')
                dt.tick_params(axis='x', colors='c')

                dt.grid(b=True, which='major', color='black', linestyle='--')
                dt.grid(b=True, which='minor', color='grey', linestyle=':')
            else:
                pass

            if 'PEF' in logs:
                pef = ax[4].twiny()
                pef.plot(data.PEF, data.DEPTH, '--', linewidth=1, label="b/elc", color='lime')
                pef.spines['top'].set_position(('outward', 60))
                pef.spines['top'].set_color('lime')
                pef.set_xlabel("PEF", color='lime')
                pef.tick_params(axis='x', colors='lime')

                pef.grid(b=True, which='major', color='black', linestyle='--')
                pef.grid(b=True, which='minor', color='grey', linestyle=':')
            else:
                pass

        # 6th track: vsh_larionov, vsh_linear
        if 'vsh_linear' and 'vsh_larionov' not in logs:
            fig.delaxes(ax=ax[5])
        else:
            ax[5].minorticks_on()
            ax[5].grid(b=True, which='major', color='black', linestyle='--')
            ax[5].grid(b=True, which='minor', color='grey', linestyle=':')

            if 'vsh_linear' in logs:
                vsh_linear = ax[5].twiny()
                vsh_linear.minorticks_on()
                vsh_linear.plot(data.vsh_linear, data.DEPTH, label='CALI[in]', color='k')
                vsh_linear.spines['top'].set_position(('outward', 20))
                vsh_linear.spines['top'].set_color('k')
                vsh_linear.set_xlabel('vsh_linear[%]', color='k')
                vsh_linear.tick_params(axis='x', colors='k')

                vsh_linear.grid(b=True, which='major', color='black', linestyle='--')
                vsh_linear.grid(b=True, which='minor', color='grey', linestyle=':')
            else:
                pass

            if 'vsh_larionov' in logs:
                vsh_larionov = ax[5].twiny()
                vsh_larionov.minorticks_on()
                vsh_larionov.plot(data.vsh_larionov, data.DEPTH, label='BS[in]', color='brown')
                vsh_larionov.spines['top'].set_position(('outward', 60))
                vsh_larionov.spines['top'].set_color('brown')
                vsh_larionov.set_xlabel('vsh_larionov[%]', color='brown')
                vsh_larionov.tick_params(axis='x', colors='brown')

                vsh_larionov.grid(b=True, which='major', color='black', linestyle='--')
                vsh_larionov.grid(b=True, which='minor', color='grey', linestyle=':')
            else:
                pass
            if 'PHIE' in logs:
                phie = ax[5].twiny()
                phie.minorticks_on()
                phie.plot(data.PHIE, data.DEPTH, linewidth=1, label='%', color='indigo')
                phie.set_xlim(1, 0.0)
                phie.spines['top'].set_position(('outward', 100))
                phie.spines['top'].set_color('indigo')
                phie.set_xlabel('PHIE [%]', color='indigo')
                phie.tick_params(axis='x', colors='indigo')

            else:
                pass

        plt.tight_layout(pad=2, h_pad=10, w_pad=2)

    @staticmethod
    def triple_combo_plot(data, min_depth: float or int or None, max_depth: float or int or None,
                          plotsize: tuple = None):
        """
        This gives the 'sp-gr', 'resistivity' and 'neutron-density' log plot

        :param data: well logs dataframe
        :param min_depth: top of reservoir, closer to the surface (length units)
        :param max_depth: bottom of reservoir, closer to the subsurface (length units)
        :param plotsize: the plot figsize in tuple form, default is (14,22)
        """
        if plotsize is None:
            plotsize = (12, 15)
        logs = data.columns
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=plotsize, sharey='all')
        fig.suptitle("Triple-Combo Plot", fontsize=22, y=1.02)

        # General setting for all axis
        for axes in ax:
            axes.get_xaxis().set_visible(False)
            axes.invert_yaxis()
            axes.spines['left'].set_color('k')
            axes.spines['right'].set_color('k')

            if min_depth is not None:
                axes.set_ylim(max_depth, min_depth)
            else:
                axes.set_ylim(data['DEPTH'].max(), data['DEPTH'].min())

        # 1st track: GR, SP
        ax[0].minorticks_on()
        ax[0].grid(b=True, which='major', color='black', linestyle='--')
        ax[0].grid(b=True, which='minor', color='grey', linestyle=':')

        # 1st track: GR, SP
        if 'GR' in logs:
            gr = ax[0].twiny()
            gr.minorticks_on()
            gr.set_xlim(0, 150)
            gr.plot(data.GR, data.DEPTH, label='GR[api]', color='green')
            gr.spines['top'].set_position(('outward', 20))
            gr.spines['top'].set_color('g')
            gr.set_xlabel('GR[api]', color='green')
            gr.tick_params(axis='x', colors='green')

            gr.grid(b=True, which='major', color='k', linestyle='--')
            gr.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'SP' in logs:
            sp = ax[0].twiny()
            sp.minorticks_on()
            sp.set_xlim(data['SP'].min(), data.max())
            sp.plot(data.GR, data.DEPTH, label='SP[mV]', color='b')
            sp.spines['top'].set_position(('outward', 60))
            sp.spines['top'].set_color('b')
            sp.set_xlabel('GR[api]', color='b')
            sp.tick_params(axis='x', colors='b')

            sp.grid(b=True, which='major', color='k', linestyle='--')
            sp.grid(b=True, which='minor', color='grey', linestyle=':')

        # 2nd track: resistivity track
        ax[1].minorticks_on()
        ax[1].grid(b=True, which='major', color='black', linestyle='--')
        ax[1].grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RD' in logs:
            rd = ax[1].twiny()
            rd.set_xlim(0.2, 2500)
            rd.spines['top'].set_position(('outward', 10))
            rd.spines['top'].set_color('y')
            rd.semilogx(data.RD, data.DEPTH, '--', linewidth=1, c='y')
            rd.set_xlabel('RD [ohm.m]', color='y')
            rd.tick_params(axis='x', colors='y')
            rd.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rd.grid(b=True, which='major', color='black', linestyle='--')
            rd.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RS' in logs:
            rs = ax[1].twiny()
            rs.set_xlim(0.2, 2500)
            rs.spines['top'].set_position(('outward', 50))
            rs.spines['top'].set_color('m')
            rs.semilogx(data.RS, data.DEPTH, linewidth=1, c='m', )
            rs.set_xlabel('RS [ohm.m]', color='m')
            rs.tick_params(axis='x', colors='m')
            rs.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rs.grid(b=True, which='major', color='black', linestyle='--')
            rs.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RM' in logs:
            rm = ax[1].twiny()
            rm.set_xlim(0.2, 2500)
            rm.spines['top'].set_position(('outward', 90))
            rm.spines['top'].set_color('C1')
            rm.semilogx(data.RM, data.DEPTH, linewidth=1, c='C1', )
            rm.set_xlabel('RM [ohm.m]', color='C1')
            rm.tick_params(axis='x', colors='C1')
            rm.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            rm.grid(b=True, which='major', color='black', linestyle='--')
            rm.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RT' in logs:
            rt = ax[1].twiny()
            rt.set_xlim(0.2, 2500)
            rt.spines['top'].set_position(('outward', 130))
            rt.spines['top'].set_color('brown')
            rt.semilogx(data.RT, data.DEPTH, linewidth=1, c='brown', )
            rt.set_xlabel('RT [ohm.m]', color='brown')
            rt.tick_params(axis='x', colors='brown')
            rt.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))

        if 'RXO' in logs:
            rxo = ax[1].twiny()
            rxo.set_xlim(0.2, 2500)
            rxo.spines['top'].set_position(('outward', 170))
            rxo.spines['top'].set_color('c')
            rxo.semilogx(data.RXO, data.DEPTH, linewidth=1, c='c', )
            rxo.set_xlabel('RXO [ohm.m]', color='c')
            rxo.tick_params(axis='x', colors='c')
            rxo.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))

        # 3rd track NPHI, DPHI, RHOB
        ax[2].minorticks_on()
        ax[2].grid(b=True, which='major', color='black', linestyle='--')
        ax[2].grid(b=True, which='minor', color='grey', linestyle=':')

        if 'NPHI' in logs:
            nphi = ax[2].twiny()
            nphi.minorticks_on()
            nphi.set_xlim(0.45, -0.15)
            nphi.spines['top'].set_position(('outward', 20))
            nphi.spines['top'].set_color('blue')
            nphi.set_xlabel("v/v")
            nphi.plot(data.NPHI, data.DEPTH, linewidth=1, label='v/v', color='blue')
            nphi.set_xlabel('NPHI [v/v] ', color='blue')
            nphi.tick_params(axis='x', colors='blue')
            nphi.xaxis.set_major_locator(plt.MultipleLocator(0.2))

            nphi.grid(b=True, which='major', color='black', linestyle='--')
            nphi.grid(b=True, which='minor', color='grey', linestyle=':')

        if 'RHOB' in logs:
            rhob = ax[2].twiny()
            rhob.set_xlim(1.95, 2.95)
            rhob.plot(data.RHOB, data.DEPTH, '--', linewidth=1, label='g/cm^3', color='red')
            rhob.spines['top'].set_position(('outward', 60))
            rhob.spines['top'].set_color('red')
            rhob.set_xlabel('RHOB [g/cm^3]', color='red')
            rhob.tick_params(axis='x', colors='red')
            rhob.xaxis.set_major_locator(plt.MultipleLocator(0.4))

        elif 'DPHI' in logs:
            dphi = ax[2].twiny()
            dphi.set_xlim(0.45, -0.15)
            dphi.plot(data.DPHI, data.DEPTH, '--', linewidth=1, label='%', color='red')
            dphi.spines['top'].set_position(('outward', 60))
            dphi.spines['top'].set_color('red')
            dphi.set_xlabel('DPHI [%]', color='red')
            dphi.tick_params(axis='x', colors='red')
            dphi.xaxis.set_major_locator(plt.MultipleLocator(0.4))

        if 'NPHI' and 'RHOB' in logs:
            # https://stackoverflow.com/questions/57766457/how-to-plot-fill-betweenx-to-fill-the-area-between-y1-and-y2-with-different-scal
            x2p, _ = (rhob.transData + nphi.transData.inverted()).transform(np.c_[data.RHOB, data.DEPTH]).T
            nphi.autoscale(False)
            nphi.fill_betweenx(data.DEPTH, data.NPHI, x2p, color="y", alpha=0.4, where=(x2p > data.NPHI))
            nphi.fill_betweenx(data.DEPTH, data.NPHI, x2p, color="turquoise", alpha=0.4, where=(x2p < data.NPHI))

        plt.tight_layout(pad=2, h_pad=10, w_pad=2)

    @staticmethod
    def doub_logplot(data, logs: list, reslog: list or None = None, min_depth: float or int or None = None,
                     max_depth: float or int or None = None, plotsize: tuple = None):
        """
        This gives a combination plot of your choice

        :param logs: name of logs to plot in a list
        :param reslog: name of resistivity logs to plot in a list
        :param data: well logs dataframe
        :param min_depth: top of reservoir, closer to the surface (length units)
        :param max_depth: bottom of reservoir, closer to the subsurface (length units)
        :param plotsize: the plot figsize in tuple form, default is (14, 22)
        """
        if plotsize is None:
            plotsize = (17, 15)

        if reslog is None:
            reslog = []

        total = len(logs) + len(reslog)
        total_logs = logs + reslog
        # create the subplots; ncols equals the number of logs
        fig, ax = plt.subplots(nrows=1, ncols=total, figsize=plotsize)

        # General setting for all axis
        for axes in ax:
            axes.invert_yaxis()
            for xtick in axes.get_xticklabels():
                xtick.set_fontsize(10)
            for ytick in axes.get_yticklabels():
                ytick.set_fontsize(10)

            if min_depth and max_depth is not None:
                axes.set_ylim(max_depth, min_depth)
            else:
                axes.set_ylim(data['DEPTH'].max(), data['DEPTH'].min())

        colors = ['k', 'brown', 'r', 'purple', 'y', 'orange', 'c', 'gold', 'b',
                  'plum', 'navy', 'm', 'sienna', 'teal', 'g']

        for i in range(len(total_logs)):
            if i < len(logs):
                # for non-resistivity, normal plot
                colrs = np.random.choice(colors)
                ax[i].minorticks_on()
                ax[i].grid(b=True, which='major', color='black', linestyle='--')
                ax[i].grid(b=True, which='minor', color='grey', linestyle=':')
                ax[i].plot(data[total_logs[i]], data['DEPTH'], color=colrs)
                ax[i].set_xlim(data[total_logs[i]].min(), data[total_logs[i]].max())
                ax[i].set_title(total_logs[i], size=20)
                ax[i].grid(b=True, which='major', color='black', linestyle='--')
                ax[i].grid(b=True, which='minor', color='grey', linestyle=':')
                colors.remove(colrs)
                if logs[i] == 'NPHI':
                    ax[i].set_xlim(0.45, -0.15)

            else:
                # for resistivity, semilog plot
                colrs = np.random.choice(colors)
                ax[i].minorticks_on()
                ax[i].grid(b=True, which='major', color='black', linestyle='--')
                ax[i].grid(b=True, which='minor', color='grey', linestyle=':')
                ax[i].semilogx(data[total_logs[i]], data['DEPTH'], color=colrs)
                ax[i].set_xlim(0.2, 2500)
                ax[i].set_title(total_logs[i], size=20)
                ax[i].grid(b=True, which='major', color='black', linestyle='--')
                ax[i].grid(b=True, which='minor', color='grey', linestyle=':')
                colors.remove(colrs)

        plt.tight_layout(1.1)


