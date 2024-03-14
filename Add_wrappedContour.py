# Load public modules 
import bpy
import bmesh
import os
import sys
import pandas as pd
import numpy as np
from math import sin, cos, asin, acos, atan2, sqrt, radians, tan


# Load local modules
dir_src = r"C:\Users\nbt571\Documents\Blender\Scripts"
dir_dep = r"C:\Users\nbt571\Documents\Blender\Scripts\dependencies"
for dir in [dir_src, dir_dep]:
    if not dir in sys.path:
        sys.path.append(dir)

import spherical_functions
import importlib
importlib.reload(spherical_functions)
from spherical_functions import sph2cart, cart2sph

import intersect
import importlib
importlib.reload(intersect)
from intersect import orientation_slope, orientation_slope_df, points_InPolygon, polysegments_inPolygon


def scale_to_radius(vctr, radius=1.0):
    """ Scales a vector to the . """
    return vctr * radius / np.linalg.norm(vctr)

def vectorial_distance(vector1, vector2):
    """ Calculates the vectorial distance between two vectors. """
    result_vector = vector1 + vector2
    return np.abs(np.linalg.norm(result_vector))

def geodesic_distance(point1Lon, point1Lat, point2Lon, point2Lat, radius):
    """ Calculates the geodesic distance between two point on a sphere,
    based on the Vincenty inverse problem formula. """
    
    # turn input coordinates from sph to radians
    lat1, lon1 = radians(point1Lat), radians(point1Lon)
    lat2, lon2 = radians(point2Lat), radians(point2Lon)

    a = cos(lat2)*sin(abs(lon2 - lon1))
    b = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(abs(lon2 - lon1))
    c = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(abs(lon2 - lon1))
    
    # geodetic distance in meters
    return radius * atan2( sqrt(a*a+b*b),c ) * 1000 

def intersect_longitude(lat1, lon1, lat2, lon2, circle_lat, radius):
    # Un-tested behaiviour if points crossed a 180 degrees longitude or
    # zero degrees latitude.
    
    if abs(lat1) < abs(lat2):       # Silly fix to make the second pnt always the one closer to the equator (it breaks otherwise, not sure why)
        point1 = sph2cart(lon2, lat2, om=1.)
        point2 = sph2cart(lon1, lat1, om=1.)
    else:    
        point2 = sph2cart(lon2, lat2, om=1.)
        point1 = sph2cart(lon1, lat1, om=1.)
    
    u_vctr = np.cross(np.array(point1), np.array(point2))
    if circle_lat == 0:
        h_vctr = np.array([0.0, 0.0, -1.0]) 
    else:
        h_vctr = np.array([0.0, 0.0, np.sign(circle_lat)]) 
    w_vctr = np.cross(h_vctr, u_vctr) 
    m_vctr = np.cross(u_vctr, w_vctr) 
        
    lon_w, _, _ = cart2sph(w_vctr[0], w_vctr[1], w_vctr[2], degreesFormat=False)
        
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
        return angle_radians
    
    
    alpha_angle = angle_between_vectors(m_vctr, h_vctr)
    
    h_length = radius * sin(radians(circle_lat))
    r_length = radius * cos(radians(circle_lat))
    p_length = h_length * tan(alpha_angle)
    gamma_angle = acos(np.sign(circle_lat) * p_length/r_length)
    

    if np.sign(circle_lat) < 0.0:
        intersection_lon = np.degrees(lon_w + np.sign(u_vctr[2]) * (np.radians(90) + gamma_angle))

    else:
        intersection_lon = np.degrees(lon_w + np.sign(u_vctr[2]) * (gamma_angle - np.radians(270)))


    normalized_longitude = (intersection_lon + 180) % 360 - 180
    
    return normalized_longitude


def wrappedContour(cntr_sph_df, sphere_obj, radius):

    # Create emtpy lists
    verts = []
    edges = []
    faces = []
    
    
    # Gather sphere unique lons and lats
    mesh = sphere_obj.data 
    sphere_vertices_all = pd.DataFrame(columns=['x', 'y', 'z', 'lat', 'lon'])
    for edge in mesh.edges:
        for vertex in [0,1]:
            new_vert = {
                'x': mesh.vertices[edge.vertices[vertex]].co.x,
                'y': mesh.vertices[edge.vertices[vertex]].co.y,
                'z': mesh.vertices[edge.vertices[vertex]].co.z,}
            sphere_vertices_all = pd.concat([sphere_vertices_all, pd.DataFrame([new_vert])], ignore_index=True)
    
    sphere_vertices_all['lon'], sphere_vertices_all['lat'], _ = cart2sph(xyz=sphere_vertices_all)
    sphere_vertices_all["lon"] = sphere_vertices_all["lon"].round(decimals=3)
    sphere_vertices_all["lat"] = sphere_vertices_all["lat"].round(decimals=3)
    unique_lons = np.sort(sphere_vertices_all['lon'].unique())
    unique_lats = np.sort(sphere_vertices_all['lat'].unique())
    
    
    #np.savetxt(r"C:\Users\nbt571\Documents\Blender\Scripts\test_outputs\results_unique_lats.txt", unique_lats, fmt="%.4f", header="lats", comments="")
    #np.savetxt(r"C:\Users\nbt571\Documents\Blender\Scripts\test_outputs\results_unique_lons.txt", unique_lons, fmt="%.4f", header="lons", comments="")


    
    # Polygon points
    poly_points = cntr_sph_df.to_numpy()
    poly_points_cart = sph2cart(poly_points[:,0], poly_points[:,1], radius)
    poly_xdeg, poly_ydeg, _ = cart2sph(poly_points_cart[0], poly_points_cart[1], poly_points_cart[2])
    poly_points_idxs = np.arange(0, len(poly_points))

    
    # Add all points plus intersecting verts
    poly_points_all = pd.DataFrame(columns=['x', 'y', 'z', 'lon', 'lat','idx_lon','idx_lat'])
    pnt_idx_count = 0
        
    for idx in poly_points_idxs:
        
        point_lon, point_lat = poly_xdeg[idx], poly_ydeg[idx]
        point_x, point_y, point_z = poly_points_cart[0][idx], poly_points_cart[1][idx], poly_points_cart[2][idx]
        
        # Classify point in face
        idx_lon = [i for i in range(len(unique_lons)) if point_lon > unique_lons[i]][-1]
        idx_lat = [i for i in range(len(unique_lats)) if point_lat > unique_lats[i]][-1]
        
        if len(poly_points_all) != 0:
        
            # If crossed both a latitude and longitude boundary
            if idx_lat != poly_points_all['idx_lat'][pnt_idx_count-1] and idx_lon != poly_points_all['idx_lon'][pnt_idx_count-1]:
                
                # Find common latitude between current and last point
                idx_lat_past = poly_points_all['idx_lat'][pnt_idx_count-1]
                lats_past = unique_lats[idx_lat_past], unique_lats[idx_lat_past+1]
                lats_current = unique_lats[idx_lat], unique_lats[idx_lat+1]
                lat_common = np.intersect1d(lats_past, lats_current)[0]
                
                 # Find common longitude between current and last point
                idx_lon_past = poly_points_all['idx_lon'][pnt_idx_count-1]
                lons_past = unique_lons[idx_lon_past], unique_lons[idx_lon_past+1]
                lons_current = unique_lons[idx_lon], unique_lons[idx_lon+1]
                lon_common = np.intersect1d(lons_past, lons_current)[0]
                sign = idx_lon_past - idx_lon
                
                # Create longitude (point crossing common latitude)
                new_point_lon_1 = intersect_longitude(poly_points_all['lat'].iloc[pnt_idx_count-1], 
                                                      poly_points_all['lon'].iloc[pnt_idx_count-1],
                                                      point_lat, point_lon, lat_common, radius)

                
                new_point_vctr_1 = sph2cart(new_point_lon_1, lat_common, radius)
                
                # Create latitude (point crossing common longitude)
                point_past = [poly_points_all['x'].iloc[pnt_idx_count-1], 
                              poly_points_all['y'].iloc[pnt_idx_count-1], 
                              poly_points_all['z'].iloc[pnt_idx_count-1]]
                
                point_current = point_x, point_y, point_z
                points_ort_vctr = np.cross(np.array(point_past), np.array(point_current))
                
                lon_ort_vctr = np.array(sph2cart(lon_common + sign*90, 0))
                new_point_vctr_2 = np.cross(points_ort_vctr, lon_ort_vctr)
                new_point_vctr_2 = scale_to_radius(new_point_vctr_2, radius)
                _, new_point_lat_2, _ = cart2sph(new_point_vctr_2[0], new_point_vctr_2[1], new_point_vctr_2[2])   

                
                dist1 = vectorial_distance(new_point_vctr_1, np.array([point_x, point_y, point_z]))
                dist2 = vectorial_distance(new_point_vctr_2, np.array([point_x, point_y, point_z]))
                
                if dist1 > dist2:
                    new_vert1 = {
                        'x': new_point_vctr_2[0],
                        'y': new_point_vctr_2[1],
                        'z': new_point_vctr_2[2],
                        'lon': lon_common,
                        'lat': new_point_lat_2,
                        'idx_lon': idx_lon_past,
                        'idx_lat': idx_lat_past,
                        }
                    new_vert2 = {
                        'x': new_point_vctr_2[0],
                        'y': new_point_vctr_2[1],
                        'z': new_point_vctr_2[2],
                        'lon': lon_common,
                        'lat': new_point_lat_2,
                        'idx_lon': idx_lon,
                        'idx_lat': idx_lat_past,
                        }          
                    new_vert3 = {
                        'x': new_point_vctr_1[0],
                        'y': new_point_vctr_1[1],
                        'z': new_point_vctr_1[2],
                        'lon': new_point_lon_1,
                        'lat': lat_common,
                        'idx_lon': idx_lon,
                        'idx_lat': idx_lat_past
                        }
                    new_vert4 = {
                        'x': new_point_vctr_1[0],
                        'y': new_point_vctr_1[1],
                        'z': new_point_vctr_1[2],
                        'lon': new_point_lon_1,
                        'lat': lat_common,
                        'idx_lon': idx_lon,
                        'idx_lat': idx_lat
                        }
                    
                else:
                    new_vert1 = {
                        'x': new_point_vctr_1[0],
                        'y': new_point_vctr_1[1],
                        'z': new_point_vctr_1[2],
                        'lon': new_point_lon_1,
                        'lat': lat_common,
                        'idx_lon': idx_lon_past,
                        'idx_lat': idx_lat_past
                        }
                    new_vert2 = {
                        'x': new_point_vctr_1[0],
                        'y': new_point_vctr_1[1],
                        'z': new_point_vctr_1[2],
                        'lon': new_point_lon_1,
                        'lat': lat_common,
                        'idx_lon': idx_lon_past,
                        'idx_lat': idx_lat
                        }
                    new_vert3 = {
                        'x': new_point_vctr_2[0],
                        'y': new_point_vctr_2[1],
                        'z': new_point_vctr_2[2],
                        'lon': lon_common,
                        'lat': new_point_lat_2,
                        'idx_lon': idx_lon_past,
                        'idx_lat': idx_lat,
                        }
                    new_vert4 = {
                        'x': new_point_vctr_2[0],
                        'y': new_point_vctr_2[1],
                        'z': new_point_vctr_2[2],
                        'lon': lon_common,
                        'lat': new_point_lat_2,
                        'idx_lon': idx_lon,
                        'idx_lat': idx_lat,
                        }   
                               
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert1])], ignore_index=True)
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert2])], ignore_index=True)
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert3])], ignore_index=True)
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert4])], ignore_index=True)
                pnt_idx_count += 4
                    
                
            # If crossed only a latitude boundary
            elif idx_lat != poly_points_all['idx_lat'][pnt_idx_count-1]:
                
                # Find common latitude between current and last point
                idx_lat_past = poly_points_all['idx_lat'][pnt_idx_count-1]
                lats_past = unique_lats[idx_lat_past], unique_lats[idx_lat_past+1]
                lats_current = unique_lats[idx_lat], unique_lats[idx_lat+1]
                lat_common = np.intersect1d(lats_past, lats_current)[0]
                
                # Create new point (not ideal, but cannot do better right now)
                avg_lon = np.mean([poly_points_all['lon'].iloc[pnt_idx_count-1], point_lon])
                new_point_lon = avg_lon
                new_point_lat = lat_common
                new_point_vctr = sph2cart(new_point_lon, new_point_lat, radius)
                
                new_vert1 = {
                    'x': new_point_vctr[0],
                    'y': new_point_vctr[1],
                    'z': new_point_vctr[2],
                    'lon': new_point_lon,
                    'lat': new_point_lat,
                    'idx_lon': idx_lon,
                    'idx_lat': idx_lat_past
                    }                
                new_vert2 = {
                    'x': new_point_vctr[0],
                    'y': new_point_vctr[1],
                    'z': new_point_vctr[2],
                    'lon': new_point_lon,
                    'lat': new_point_lat,
                    'idx_lon': idx_lon,
                    'idx_lat': idx_lat
                    }
                
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert1])], ignore_index=True)
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert2])], ignore_index=True)
                pnt_idx_count += 2
                
            # If crossed only a longitude boundary
            elif idx_lon != poly_points_all['idx_lon'][pnt_idx_count-1]:
                
                # Find common longitude between current and last point
                idx_lon_past = poly_points_all['idx_lon'][pnt_idx_count-1]
                lons_past = unique_lons[idx_lon_past], unique_lons[idx_lon_past+1]
                lons_current = unique_lons[idx_lon], unique_lons[idx_lon+1]
                lon_common = np.intersect1d(lons_past, lons_current)[0]
                sign = idx_lon_past - idx_lon
                
                # Create new point
                point_past = [poly_points_all['x'].iloc[pnt_idx_count-1], 
                              poly_points_all['y'].iloc[pnt_idx_count-1], 
                              poly_points_all['z'].iloc[pnt_idx_count-1]]
                point_current = point_x, point_y, point_z
                points_ort_vctr = np.cross(np.array(point_past), np.array(point_current))
                lon_ort_vctr = np.array(sph2cart(lon_common + sign*90, 0))
                new_point_vctr = np.cross(points_ort_vctr, lon_ort_vctr)
                new_point_vctr = scale_to_radius(new_point_vctr, radius)
                new_point_lon, new_point_lat, _ = cart2sph(new_point_vctr[0], new_point_vctr[1], new_point_vctr[2])
                
                new_vert1 = {
                    'x': new_point_vctr[0],
                    'y': new_point_vctr[1],
                    'z': new_point_vctr[2],
                    'lon': new_point_lon,
                    'lat': new_point_lat,
                    'idx_lon': idx_lon_past,
                    'idx_lat': idx_lat
                    }
                new_vert2 = {
                    'x': new_point_vctr[0],
                    'y': new_point_vctr[1],
                    'z': new_point_vctr[2],
                    'lon': new_point_lon,
                    'lat': new_point_lat,
                    'idx_lon': idx_lon,
                    'idx_lat': idx_lat
                    }
                
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert1])], ignore_index=True)
                poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert2])], ignore_index=True)
                pnt_idx_count += 2
                
        new_vert_scaled = scale_to_radius(np.array([point_x, point_y, point_z]), radius)
        new_vert = {
            'x': new_vert_scaled[0],
            'y': new_vert_scaled[1],
            'z': new_vert_scaled[2],
            'lon': point_lon,
            'lat': point_lat,
            'idx_lon':idx_lon,
            'idx_lat':idx_lat}
        poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert])], ignore_index=True)
        pnt_idx_count += 1
        
    
    # Edges
    edges = [(i, i+1) for i in range(pnt_idx_count-1)]
    
    

        
    def assign_next_idx(df, idx):
        if idx in df.index:
            idx = idx+1
            return assign_next_idx(df, idx)
        else: return idx

    # Faces
    def eval_last_point(poly_points_all, radius, current_face_idf_tosplit=[], current_face_idf_topass=[], face_done=[]): 
        
        df_chunks = []
        if current_face_idf_tosplit:
            # Find the indices where the monotonic increase occurs
            split_indices = np.where(np.diff(current_face_idf_tosplit[1].index) != 1)[0] + 1
            
            # Split the DataFrame into chunks based on monotonic increase
            df_chunks = np.split(current_face_idf_tosplit[1], split_indices)
            
        if current_face_idf_topass:
            df_chunks.extend(current_face_idf_topass)

       
        n_chunks = len(df_chunks)
        
        while n_chunks > 0:
            
            current_face_df = df_chunks[0]
            
                
            remaining_chunks = [df for i, df in enumerate(df_chunks) if i != 0]
            
            p = current_face_df.iloc[-2]
            q = current_face_df.iloc[-1]
            
            # End points in same longitude boundary
            if current_face_df['lon'].iloc[0] == current_face_df['lon'].iloc[-1]:
                r = current_face_df.iloc[0]
                if orientation_slope_df(p, q, r) == 2:
                    face_done.append(current_face_df)
                    n_chunks -= 1
                    df_chunks = remaining_chunks
                    continue
                    
            
            # End points in same latitude boundary
            if current_face_df['lat'].iloc[0] == current_face_df['lat'].iloc[-1]:
                r = current_face_df.iloc[0]
                if orientation_slope_df(p, q, r) == 2:
                    face_done.append(current_face_df)
                    n_chunks -= 1
                    df_chunks = remaining_chunks
                    continue
            
            
            # Last point in same boundary as other chunk's first point
            connecting_chunks_idx = [i for i, df in enumerate(remaining_chunks) 
                                     if current_face_df['lat'].iloc[-1] == df.iloc[0]['lat'] 
                                     or current_face_df['lon'].iloc[-1] == df.iloc[0]['lon']
                                     ]

            
            if len(connecting_chunks_idx) == 1:
                r = remaining_chunks[connecting_chunks_idx[0]].iloc[0]
                if orientation_slope_df(p, q, r) == 2:
                    current_face_df = pd.concat([current_face_df, remaining_chunks[connecting_chunks_idx[0]]])
                    remaining_chunks = [df for i, df in enumerate(remaining_chunks) if i not in connecting_chunks_idx]
                    current_face_idf_topass = [current_face_df]
                    current_face_idf_topass.extend(remaining_chunks)
                    return eval_last_point(poly_points_all, radius, [], current_face_idf_topass, face_done)
            
            elif len(connecting_chunks_idx) > 1:
                
                # ADD ORIENTATION ASSESTMENT!
                
                # Find smallest distance
                distance = []
                for idx_c in connecting_chunks_idx:
                    distance.append(geodesic_distance(current_face_df.iloc[-1]["lon"], 
                                                     current_face_df.iloc[-1]["lat"], 
                                                     remaining_chunks[idx_c].iloc[0]["lon"], 
                                                     remaining_chunks[idx_c].iloc[0]["lat"], 
                                                     radius))
                
                minpos = distance.index(min(distance))
                current_face_df = pd.concat([current_face_df, remaining_chunks[minpos]])
                remaining_chunks = [df for i, df in enumerate(remaining_chunks) if i != minpos]
                current_face_idf_topass = [current_face_df]
                current_face_idf_topass.extend(remaining_chunks)
                return eval_last_point(poly_points_all, radius, [], current_face_idf_topass, face_done)
        
        
            
            # End points are not on the same edge
            stlast_row = current_face_df.iloc[-2]
            last_row = current_face_df.iloc[-1]
            p = stlast_row['lon'], stlast_row['lat']
            q = last_row['lon'], last_row['lat']
            
            idx_lon = last_row['idx_lon']
            idx_lat = last_row['idx_lat']
            
            # If last point is in latitude boundary
            if last_row['lat'] in list(unique_lats):
                # Find sphere face segment
                lons_edges = [unique_lons[idx_lon], unique_lons[idx_lon-1], unique_lons[idx_lon+1]]  # problematic when it comes to 360-0
                orients = [orientation_slope(p, q, [le, last_row['lat']]) for le in lons_edges]
                try:
                    new_vert_sph = lons_edges[orients.index(2)], last_row['lat']
                except ValueError:
                    pass
               
            # If last point is in longitude boundary
            if last_row['lon'] in list(unique_lons):
                # Find sphere face segment
                lats_edges = [unique_lats[idx_lat], unique_lats[idx_lat-1], unique_lats[idx_lat+1]]  # problematic when it comes to 360-0
                orients = [orientation_slope(p, q, [last_row['lon'], le]) for le in lats_edges]
                try:
                    new_vert_sph = last_row['lon'], lats_edges[orients.index(2)]
                except ValueError:
                    pass
        
                
            new_vert_cart = sph2cart(new_vert_sph[0], new_vert_sph[1], radius)
            new_vert = {
                'x': new_vert_cart[0],
                'y': new_vert_cart[1],
                'z': new_vert_cart[2],
                'lon': new_vert_sph[0],
                'lat': new_vert_sph[1],
                'idx_lon': last_row['idx_lon'],
                'idx_lat': last_row['idx_lat'],
                }
            
            idx_poly_new = assign_next_idx(current_face_df, len(poly_points_all))
            if len(face_done) > 0:
                idx_poly_new = assign_next_idx(pd.concat(face_done), idx_poly_new)
            current_face_df = pd.concat([current_face_df, pd.DataFrame([new_vert], index=[idx_poly_new])])
            current_face_idf_topass = [current_face_df]
            current_face_idf_topass.extend(remaining_chunks)
            return eval_last_point(poly_points_all, radius, [], current_face_idf_topass, face_done)
        
        return face_done 

        
    # Group boundary segments by face
    contour_faces_list = list(poly_points_all.groupby(["idx_lon", "idx_lat"]))   

    for face_n, face_df in enumerate(contour_faces_list):
         
        face_df_topass = []
        face_df_tosplit = face_df
        if face_df[1].iloc[0].name == 0:
            if poly_points_all.iloc[0]["idx_lon"] == poly_points_all.iloc[pnt_idx_count-1]["idx_lon"] and poly_points_all.iloc[0]["idx_lat"] == poly_points_all.iloc[pnt_idx_count-1]["idx_lat"]:

                first_df = face_df
                contour_faces_list = [idf for i, idf in enumerate(contour_faces_list) if i != first_df[0]]
                
                # Split the DataFrame into chunks based on monotonic increase
                split_indices = np.where(np.diff(first_df[1].index) != 1)[0] + 1
                df_chunks = np.split(first_df[1], split_indices)
                
                # Append first chunk and other chunks separately
                df_first_chunk = [df for df in df_chunks if df.iloc[0].name == 0][0]
                df_last_chunk = [df for df in df_chunks if df.iloc[-1].name == pnt_idx_count-1][0]
                df_first_chunk = pd.concat([df_last_chunk, df_first_chunk])
                face_df_topass.append(df_first_chunk)
                
                df_other_chunks = [df for df in df_chunks if df.iloc[0].name != 0 and df.iloc[-1].name != pnt_idx_count-1]
                if len(df_other_chunks) > 0:
                    df_other_chunks = pd.concat(df_other_chunks)
                    face_df_tosplit = (first_df[0], df_other_chunks)
                else:
                    face_df_tosplit = []
        
        # Close boundary segments into polygons
        current_new_face_list = eval_last_point(poly_points_all, radius, face_df_tosplit, face_df_topass, [])
        
        # Add new points to poly_points_all, and index lists to face
        for current_new_face_df in current_new_face_list:
            new_points_keys = np.setdiff1d(list(current_new_face_df.index), list(face_df[1].index))
            for new_pnt_key in new_points_keys:
                poly_points_all = pd.concat([poly_points_all, current_new_face_df.loc[new_pnt_key].to_frame().T])
            
            faces.append(tuple(current_new_face_df.index[::-1]))

                

    # Iterate over all sphere faces
    for sphere_face in mesh.polygons:
        
        # Transform faces to spherical coordinates
        verts_in_face_x = [mesh.vertices[vert_index].co.x for vert_index in sphere_face.vertices]
        verts_in_face_y = [mesh.vertices[vert_index].co.y for vert_index in sphere_face.vertices]
        verts_in_face_z = [mesh.vertices[vert_index].co.z for vert_index in sphere_face.vertices]
        face_xdeg, face_ydeg,_ = cart2sph(verts_in_face_x, verts_in_face_y, verts_in_face_z)
        
        # Check if face vertices are inside polygon
        in_polygon = points_InPolygon(list(zip(poly_xdeg, poly_ydeg)), list(zip(face_xdeg, face_ydeg))) 
        
        
        # Central faces
        if all(in_polygon):
            
            # Check if face segments intersect polygon
            intersects_polygon = polysegments_inPolygon(list(zip(poly_xdeg, poly_ydeg)), list(zip(face_xdeg, face_ydeg)))
            
            if not intersects_polygon:
                verts_idx_face = []
            
                # Add verts
                for vert_index in sphere_face.vertices:
                    
                    
                    verts_idx_face.append(len(poly_points_all))
                    vert_i = mesh.vertices[vert_index].co
                    new_vert_scaled = scale_to_radius(np.array([vert_i.x, vert_i.y, vert_i.z]), radius)
                    new_vert = {
                        'x': new_vert_scaled[0],
                        'y': new_vert_scaled[1],
                        'z': new_vert_scaled[2]}
                    poly_points_all = pd.concat([poly_points_all, pd.DataFrame([new_vert])], ignore_index=True)
                
                # Add faces
                faces.append(verts_idx_face)

    verts = [[row['x'], row['y'], row['z']] for row in poly_points_all.iloc()]
         
    return verts, edges, faces