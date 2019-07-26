# import pandas as pd
#   
# # dictionary of lists
# dict = {'name':["aparna", "pankaj", "sudhir", "Geeku"],
#         'degree': ["MBA", "BCA", "M.Tech", "MBA"],
#         'score':[90, 40, 80, 98]}
#  
# # creating a dataframe from a dictionary 
# df = pd.DataFrame(dict)
# df.style
# #print(df)

import plotly.plotly as py
import plotly.graph_objs as go

trace = go.Table(
    header=dict(values=['A Scores', 'B Scores'],
                line = dict(color='#7D7F80'),
                fill = dict(color='#a1c3d1'),
                align = ['left'] * 5),
    cells=dict(values=[[100, 90, 80, 90],
                       [95, 85, 75, 95]],
               line = dict(color='#7D7F80'),
               fill = dict(color='#EDFAFF'),
               align = ['left'] * 5))

layout = dict(width=500, height=300)
data = [trace]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename = 'styled_table')