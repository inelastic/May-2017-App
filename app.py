###############################################################################
#Import libraries
###############################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
###############################################################################
#Define Functions to be used here
###############################################################################
#the function for laser off
def loff_func(x,a,b,g):
    return a + b*np.exp(-g*x)

#create sample temperature plot
def temp_plot(df):
    hovertemp ='<B>Time(s)</B>: %{x}'+'<br><b>Temperature(K)</b>: %{y}<extra></extra>'
    
    trace1 = go.Scatter(
        name='Sample Temperature',
        x=df['Time(s)'],
        y=df['Temp(K)'],
        hovertemplate=hovertemp,
        mode='lines',
        line=dict(color='rgb(31, 119, 255)'),
    )
    
    layout = go.Layout(
        xaxis=dict(title='Time(s)',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        yaxis=dict(title='Temperature(K)',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel_align = 'left',
        plot_bgcolor='rgb(255,255,255)',        
        showlegend = True)
    if (df['Temp(K)'].mean() > 0):
        fig = go.Figure(data=trace1, layout=layout)
        fig.update_layout(
            title="No Temperature Data Available",
        )
    else:
        fig=go.Figure(data=[], layout=layout)    
    return fig
    
#Create asymmetry plots. There are 3 options for display. The figures are added
#to the plot based on which options are checked.
def repol_plot(repol, repolfitdata):    
    hovertemp ='<B>Time(us)</B>: %{x}'+'<br><b>Asymmetry</b>: %{y}<extra></extra>'
    
    trace1 = go.Scatter(
        name='A',
        x=repol['B-Field(G)'],
        y=repol['Loff A'],
        hovertemplate=hovertemp,
        mode='markers',
        line=dict(color='rgb(0, 255, 0)'),
        error_y=dict(
            type='data',
            array=repol['Loff A Err'],
            visible=True,
            color='darkgray',
            thickness=1))
    
    trace2 = go.Scatter(
        name='B',
        x=repol['B-Field(G)'],
        y=repol['Loff B'],
        hovertemplate=hovertemp,        
        mode='markers',
        line=dict(color='rgb(255, 0, 0)'),
        error_y=dict(
            type='data',
            array=repol['Loff B Err'],
            visible=True,
            color='darkgray',
            thickness=1))
    
    trace3 = go.Scatter(
        name='T0 Fit',
        x=repolfitdata['B'],
        y=repolfitdata['T0y'],
        hovertemplate=hovertemp,        
        mode='lines',
        line=dict(color='rgb(255, 0, 0)'))
    
    trace4 = go.Scatter(
        name='BC0 Fit',
        x=repolfitdata['B'],
        y=repolfitdata['BCy'],
        hovertemplate=hovertemp,        
        mode='lines',
        line=dict(color='rgb(0, 255, 0)'))

    data = [trace1, trace2, trace3, trace4]
        
#create layout of plot    
    layout = go.Layout(
        xaxis=dict(title='B-Field(G)',
            mirror=True,
            range=[-1500,1500],
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        yaxis=dict(title='Asymmetry',
            mirror=True,
            range=[0,0.2],
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        width=800,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel_align = 'left',
        plot_bgcolor='rgb(255,255,255)',        
        showlegend = True)
    
    fig = go.Figure(data=data, layout=layout)
    return fig

def ds_vs_lp(data):
    hovertemp ='<B>Power Red(%)</B>: %{x}'+'<br><b>Asymmetry(%)</b>: %{y}<extra></extra>'
    
    fig1 = go.Scatter(
        name=r'$\text{B=2750 / } \sigma^{-} - \sigma^{+} \text{ / May 2017}$',x=data.loc[(data['B-Field(G)'] == 2750) & (data['QWP Pos'] == 0)]['Power Drop'].values*100,y=data.loc[(data['B-Field(G)'] == 2750) & (data['QWP Pos'] == 0)]['Asym Diff'].values*100,
        mode='markers', marker_color='rgb(200,0,0)', marker_symbol=105, marker_size=15, marker_line_width=3,
        hovertemplate=hovertemp,
        error_y=dict(
            type='data',
            array=data.loc[(data['B-Field(G)'] == 2750) & (data['QWP Pos'] == 0)]['Asym Diff Err'].values*100,
            visible=True,
            color='darkgray',
            thickness=1)
    )
    fig2 = go.Scatter(
        name=r'$\text{B=-2750 / } \sigma^{-} - \sigma^{+} \text{ / May 2017}$',x=data.loc[(data['B-Field(G)'] == -2750) & (data['QWP Pos'] == 0)]['Power Drop'].values*100,y=data.loc[(data['B-Field(G)'] == -2750) & (data['QWP Pos'] == 0)]['Asym Diff'].values*100,
        mode='markers', marker_color='rgb(0,0,0)', marker_symbol=105, marker_size=15, marker_line_width=3,
        hovertemplate=hovertemp,
        error_y=dict(
            type='data',
            array=data.loc[(data['B-Field(G)'] == -2750) & (data['QWP Pos'] == 0)]['Asym Diff Err'].values*100,
            visible=True,
            color='darkgray',
            thickness=1)   
    )
    fig3 = go.Scatter(
        name=r'$\text{B=2750 / } \sigma^{+} - \sigma^{-} \text{ / May 2017}$',x=data.loc[(data['B-Field(G)'] == 2750) & (data['QWP Pos'] == 1)]['Power Drop'].values*100,y=data.loc[(data['B-Field(G)'] == 2750) & (data['QWP Pos'] == 1)]['Asym Diff'].values*100,
        mode='markers',marker_color='rgb(200,0,0)', marker_symbol=101, marker_size=15, marker_line_width=3,
        hovertemplate=hovertemp,
        error_y=dict(
            type='data',
            array=data.loc[(data['B-Field(G)'] == 2750) & (data['QWP Pos'] == 1)]['Asym Diff Err'].values*100,
            visible=True,
            color='darkgray',
            thickness=1)       
    )
    fig4 = go.Scatter(
        name=r'$\text{B=-2750 / } \sigma^{+} - \sigma^{-} \text{ / May 2017}$',x=data.loc[(data['B-Field(G)'] == -2750) & (data['QWP Pos'] == 1)]['Power Drop'].values*100,y=data.loc[(data['B-Field(G)'] == -2750) & (data['QWP Pos'] == 1)]['Asym Diff'].values*100,
        mode='markers',marker_color='rgb(0,0,0)', marker_symbol=101, marker_size=15, marker_line_width=3,
        hovertemplate=hovertemp,
        error_y=dict(
            type='data',
            array=data.loc[(data['B-Field(G)'] == -2750) & (data['QWP Pos'] == 1)]['Asym Diff Err'].values*100,
            visible=True,
            color='darkgray',
            thickness=1)  
    )
    fig5 = go.Scatter(
        name='origin line', x=[0,70], y=[0,0],
        mode='lines', line_width=2, line_color='rgb(0,0,0)', showlegend=False)
    
    #create layout of plot    
    layout = go.Layout(
        xaxis=dict(title=r'$\Delta(\sigma^{+} + \sigma^{-}) \text{(%)}$',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        yaxis=dict(title='Asymmetry Change(%)',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        width=800,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel_align = 'left',
        plot_bgcolor='rgb(255,255,255)',        
        showlegend = True)
    
    fig = go.Figure(data=[fig1, fig2, fig3, fig4, fig5], layout=layout)


    return fig

def lam_plot(repol, repolfitdata):
    repol = repol.iloc[9:]
    hovertemp ='<B>Time(us)</B>: %{x}'+'<br><b>Asymmetry</b>: %{y}<extra></extra>'
    
    trace1 = go.Scatter(
        name='G',
        x=repol['B-Field(G)'],
        y=repol['Loff g'],
        hovertemplate=hovertemp,
        mode='markers',
        line=dict(color='rgb(255, 0, 0)'),
        error_y=dict(
            type='data',
            array=repol['Loff g Err'],
            visible=True,
            color='darkgray',
            thickness=1))    
    trace2 = go.Scatter(
        name='G Fit',
        x=repolfitdata['B'],
        y=repolfitdata['ly'],
        hovertemplate=hovertemp,        
        mode='lines',
        line=dict(color='rgb(255, 0, 255)'))
    

    data = [trace1, trace2]
        
#create layout of plot    
    layout = go.Layout(
        xaxis_type = "log",
        xaxis=dict(title='B-Field(G)',
            mirror=True,
            ticks='outside',
            showline=True,            
            linecolor='rgb(0,0,0)'),
        yaxis=dict(title='Asymmetry',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        width=800,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel_align = 'left',
        plot_bgcolor='rgb(255,255,255)',        
        showlegend = True)
    
    fig = go.Figure(data=data, layout=layout)    
    
    
    return fig

def asym_plots(df, run, df2, plots):    
    hovertemp ='<B>Time(us)</B>: %{x}'+'<br><b>Asymmetry</b>: %{y}<extra></extra>'
    cols = [col for col in df.columns if 'Laser On' in col]
    
    trace1e = go.Scatter(
        name='Laser Off',
        x=df['Time'],
        y=df['Laser Off'],
        hovertemplate=hovertemp,
        mode='lines',
        line=dict(color='rgb(31, 119, 255)'),
        error_y=dict(
            type='data',
            array=df['Laser Off Err'],
            visible=True,
            color='darkgray',
            thickness=1))
    
    
    trace1 = go.Scatter(
        name='Laser Off',
        x=df['Time'],
        y=df['Laser Off'],
        hovertemplate=hovertemp,        
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'))
    

    trace3 = go.Scatter(
        name='Laser Off Fit',
        x=df['Time'],
        y=loff_func(df['Time'],*[df2.loc[run]['Loff A'], df2.loc[run]['Loff B'], df2.loc[run]['Loff g']]),
        hovertemplate=hovertemp,        
        mode='lines',
        line=dict(color='rgb(255, 0, 0)'))
    
    if len(cols) > 0:
        trace2e = go.Scatter(
            name='Laser On',
            x=df['Time'],
            y=df['Laser On'],
            hovertemplate=hovertemp,        
            mode='lines',
            line=dict(color='rgb(0, 255, 35)'),
            error_y=dict(
                type='data',
                array=df['Laser On Err'],
                visible=True,
                color='darkgray',
                thickness=1))
        
        trace2 = go.Scatter(
            name='Laser On',
            x=df['Time'],
            y=df['Laser On'],
            hovertemplate=hovertemp,        
            mode='lines',
            line=dict(color='rgb(0, 255, 35)'))
    
        data = []
        if len(plots) == 0:
            data = [trace1]
        elif len(plots) == 1:
            if plots == ['LOF']:
                data = [trace1, trace3]
            elif plots == ['LOE']:
                data = [trace1, trace2]
            elif plots == ['EB']:
                data = [trace1e]
        elif len(plots) == 2:
            if plots == ['LOF', 'LOE']:
                data = [trace1, trace2, trace3]
            if plots == ['LOE', 'LOF']:
                data = [trace1, trace2, trace3]
            elif plots == ['LOF', 'EB']:
                data = [trace1e, trace3]
            elif plots == ['EB', 'LOF']:
                data = [trace1e, trace3]            
            elif plots == ['LOE', 'EB']:
                data = [trace1e, trace2e]
            elif plots == ['EB', 'LOE']:
                data = [trace1e, trace2e]            
        elif len(plots) == 3:
            data = [trace1e, trace2e, trace3]
            
    else:          
        data = []
        if len(plots) == 0:
            data = [trace1]
        elif len(plots) == 1:
            if plots == ['LOF']:
                data = [trace1, trace3]
            elif plots == ['LOE']:
                data = [trace1, trace2]
            elif plots == ['EB']:
                data = [trace1e]
        elif len(plots) == 2:
            if plots == ['LOF', 'LOE']:
                data = [trace1, trace3]
            if plots == ['LOE', 'LOF']:
                data = [trace1, trace3]
            elif plots == ['LOF', 'EB']:
                data = [trace1e, trace3]
            elif plots == ['EB', 'LOF']:
                data = [trace1e, trace3]            
            elif plots == ['LOE', 'EB']:
                data = [trace1e]
            elif plots == ['EB', 'LOE']:
                data = [trace1e]            
        elif len(plots) == 3:
            data = [trace1e, trace3]
        
    
        
#create layout of plot    
    layout = go.Layout(
        xaxis=dict(title='Time(us)',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        yaxis=dict(title='Asymmetry',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel_align = 'left',
        plot_bgcolor='rgb(255,255,255)',        
        showlegend = True)
    
    fig = go.Figure(data=data, layout=layout)
    return fig

def build_table(run):
    tempave = tempdata[run]['Temp(K)'].mean()
    tempaverr = tempdata[run]['Temp(K)'].sem(axis=0)
    row1 = html.Tr([html.Td("# Of Events (Mev)"), html.Td(fitdata.loc[run]['Raw Events(MeV)'])])
    if tempave > 0:
        row2 = html.Tr([html.Td("Signal (%)"), html.Td('%5.5f +- %5.5f' %(fitdata.loc[run]['Asym Diff'], abs(fitdata.loc[run]['Asym Diff Err'])))])
        row3 = html.Tr([html.Td("Asym Drop (%)"), html.Td('%5.5f +- %5.5f' %(fitdata.loc[run]['Power Drop']*100, abs(fitdata.loc[run]['Power Drop Err']*100)))])
        row4 = html.Tr([html.Td("Temp Ave (K)"), html.Td('%5.3f +- %5.5f' %(tempave, tempaverr))])
    else:
        row2 = html.Tr([html.Td("Signal (%)"), html.Td('No Laser Data')])
        row3 = html.Tr([html.Td("Asym Drop (%)"), html.Td('No Laser Data')])
        row4 = html.Tr([html.Td("Temp Ave (K)"), html.Td("No Temp Data")])
    table_body = [html.Tbody([row1, row2, row3, row4])]
    return table_body

def lsat_plot(laser_sat, laser_sat_data):
    hovertemp ='<B>Power(mJ)</B>: %{x}'+'<br><b>Asym Red</b>: %{y}<extra></extra>'
    
    trace1 = go.Scatter(
        name='Data',
        x=laser_sat_data['Ps(mJ/Pulse)'],
        y=laser_sat_data['(Lon-Loff)/Loff'],
        hovertemplate=hovertemp,
        mode='markers',
        line=dict(color='rgb(255, 0, 0)'),
        error_y=dict(
            type='data',
            array=laser_sat_data['err.2'],
            visible=True,
            color='darkgray',
            thickness=1))  
    
    trace2 = go.Scatter(
        name='Fit',
        x=laser_sat['Power'],
        y=laser_sat['Asym Red'],
        hovertemplate=hovertemp,        
        mode='lines',
        line=dict(color='rgb(255, 0, 255)'))
    
    data = [trace1, trace2]
        
#create layout of plot    
    layout = go.Layout(
        xaxis=dict(title='Power(mJ/Pulse)',
            ticks='outside',
            showline=True,            
            linecolor='rgb(0,0,0)'),
        yaxis=dict(title='Asym Red(%)',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='rgb(0,0,0)'),
        width=800,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel_align = 'left',
        plot_bgcolor='rgb(255,255,255)',        
        showlegend = True)
    
    fig = go.Figure(data=data, layout=layout)    
    
    
    return fig

    
###############################################################################
#Setup data to be displayed
###############################################################################
#setting the directory and grabbing the filenames
directory="https://raw.githubusercontent.com/inelastic/May-2017-App/master/ProcessedCSV/"
#directory="F:/Storage/Documents/Python Files/RALAnalysis/May-2017-App/ProcessedCSV/"
filenames = filenames = pd.read_csv(directory+'listfile.txt', names=['Name'])
filenames.drop(filenames.tail(1).index,inplace=True)
filenames = filenames['Name'].values
filenames = filenames.astype(int)


fitdata = pd.read_csv(directory+'fitdata.csv', index_col=0)

#re-indexing fitdata to use in graphs

#set dict to hold data
tempdata = {}
Asym = {}

wavelengths = np.empty(len(filenames), dtype=object) 
qwppos = np.zeros(len(filenames))
for i in range(len(filenames)): 
    temp1 = pd.read_csv(directory+'Sample_Temperature/'+str(filenames[i])+'.csv', index_col=0)
    temp1['Time']= pd.to_datetime(temp1['Time']) 
    name = fitdata.index[i]
    tempdata[name] = temp1
    temp2 = pd.read_csv(directory+'Asymmetry/'+str(filenames[i])+'.csv')
    try:
        wavelengths[i] = str(temp1['Wavelength'].values[0])
        qwppos[i] = str(temp1['QWP_Pos'].values[0])       
    except:
        print()
    finally:
        Asym[name] = temp2
        

#Create PvsLP 
pvslp = ds_vs_lp(fitdata)
                    
###############################################################################
#Build Dash App
###############################################################################
#create Summary table
table_header = [
    #html.Thead(html.Tr([html.Th("First Name"), html.Th("Last Name")]))
]
tempave = tempdata[filenames[0]]['Temp(K)'].mean()
tempaverr = tempdata[filenames[0]]['Temp(K)'].sem(axis=0)
table_body = build_table(filenames[0])

#create text for drop down list (file, wavelength, B-Field)
wavelengths = fitdata['Wavelength(nm)'].values.astype(str)
for i in range(len(wavelengths)):
    if wavelengths[i] == 'nan':
        wavelengths[i] = 'No Laser'


fitdata['Names'] = fitdata.index.astype(str)+", "+wavelengths+", "+ fitdata['B-Field(G)'].astype(str)
#fitdata['Names'] = fitdata.index.astype(str)+", "+fitdata['Wavelength(nm)'].astype(str)+", "+ fitdata['B-Field(G)'].astype(str)

# Build App
stylesheet = 'https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/cerulean/bootstrap.min.css'
app = dash.Dash(__name__, external_scripts=[
  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'],
  external_stylesheets=[stylesheet]
)
server = app.server


#create layout
app.layout = dbc.Container([
    html.H1("May 2017 Si Results"),
    dbc.Row(
        [
            dbc.Col(dbc.Card(
                [
                    dbc.CardHeader("Data (File, Wavelength, B-Field)"),
                    dbc.CardBody(
                        [            
                            dbc.Row(dbc.Col(dcc.Dropdown(id='wavelength-dropdown',
                                         clearable=False,
                                         options=[{'label': i, 'value': i} for i in fitdata['Names'].values],
                                         value=fitdata['Names'].values[0]
                                         )))
                        ]
                    ),                   
                ],
            ),),
        dbc.Col(dbc.Table(id="summary-table", children=table_body, bordered=True))
        ]),    
    html.Hr(),
    html.H2("Sample Temperature"),
    dcc.Graph(id='temp'),
    html.Hr(),
    html.H2('Asymmetry Plots'),
    dcc.Graph(id='asym'),
    dcc.Checklist(
        id='plots',
        options=[
            {'label': 'Laser Off Fit', 'value': 'LOF'},
            {'label': 'Laser On', 'value': 'LOE'},
            {'label': 'Error Bars', 'value': 'EB'}
        ],
    value=['LOF', 'LOE', 'EB'],
    labelStyle={'display': 'inline-block', 'margin-right': 8}
    ),
    html.Hr(),
    html.H2("Asymmetry Change vs Laser Power Asymmetry Change"),
    dcc.Graph(figure=pvslp),
  
])

# Define callbacks to update graph
@app.callback(
   [Output('temp', 'figure'),
    Output('asym','figure')],
    [Input("wavelength-dropdown", "value"),
     Input("plots", 'value')]
)
def update_figure(tempdata_value, plots):
    run = int(tempdata_value[0:6])
    temperdata = tempdata[run]
    asymdata = Asym[run]
    tempfig = temp_plot(temperdata)
    asymfig = asym_plots(asymdata, run, fitdata, plots)
    return tempfig, asymfig
    
@app.callback(
    [Output('summary-table', 'children')],
    [Input("wavelength-dropdown","value")],    
)
def update_table(wavelength):
    run = int(wavelength[0:6])    
    table_body = build_table(run)
    return table_body
    
if __name__ == '__main__':    
    app.run_server(debug=True)