# NOAA StormEvent historical weather event data

## Projects:

### Graphing the total yearly cost of weather events from 1950 to 2017

A bar chart showing the total inflation adjusted cost of recorded weather events throughout the years in the dataset.

- [JMB_storm_data_cost_vf.ipynb](http://nbviewer.jupyter.org/github/johnmburt/projects/blob/master/NOAA_weather_events/JMB_storm_data_cost_vf.ipynb)

![yearly cost](yearly_cost.png)


### Interactive map with weather event overlay 

My goal for this project was to create an interactive map plot allowing users to visualize the impact of selected weather events over space and time. The plot has a list selector for users to select different weather event types to plot, and a date slider so that they can select the time range to view. 

The visualization was generated using the interactive graphing package bokeh.

- [NOAA_weather_events/weather_map_bokeh_googlemaps_app_vf.ipynb](http://nbviewer.jupyter.org/github/johnmburt/projects/blob/master/NOAA_weather_events/weather_map_bokeh_googlemaps_app_vf.ipynb)

### Map at interactive session startup:

![NOAA weather map](./weather_event_map.png)

### Map zoomed in and date selected to show the 4/27/2011 tornado "Super Outbreak".
![NOAA weather map](./weather_event_map_04-27-11_outbreak.png)
