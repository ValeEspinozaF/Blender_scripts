# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:18:34 2023

@author: Valentina Espinoza
"""

import numpy as np
import pandas as pd
from matplotlib import path


def sph2cart_array(azimuth, elevation, r):
    # azimuth, elevation in radians

    m = np.size(elevation,0)

    x = np.zeros(m)
    y = np.zeros(m)
    z = np.zeros(m)

    for ii in range(m):
        x[ii] = r * np.cos(elevation[ii]) * np.cos(azimuth[ii])
        y[ii] = r * np.cos(elevation[ii]) * np.sin(azimuth[ii])
        z[ii] = r * np.sin(elevation[ii])

    return x, y, z



def finite_rotation_matrix(xrt, yrt, zrt, ang):
    # the angle (ang) is in degrees
    
    rot_mtx = np.array([ [np.cos(ang)+xrt**2*(1-np.cos(ang)), xrt*yrt*(1-np.cos(ang))-zrt*np.sin(ang), xrt*zrt*(1-np.cos(ang))+yrt*np.sin(ang)],
                         [yrt*xrt*(1-np.cos(ang))+zrt*np.sin(ang), np.cos(ang)+yrt**2*(1-np.cos(ang)), yrt*zrt*(1-np.cos(ang))-xrt*np.sin(ang)],
                         [zrt*xrt*(1-np.cos(ang))-yrt*np.sin(ang), zrt*yrt*(1-np.cos(ang))+xrt*np.sin(ang), np.cos(ang)+zrt**2*(1-np.cos(ang))] ])

    return rot_mtx



def points_in_polygon(polygon_xdeg, polygon_ydeg, points_xdeg, points_ydeg):
    
    poly_xrad = polygon_xdeg*(np.pi/180.)
    poly_yrad = polygon_ydeg*(np.pi/180.)
    points_xrad = points_xdeg*(np.pi/180.)
    points_yrad = points_ydeg*(np.pi/180.)
    
    xc, yc, zc = sph2cart_array(poly_xrad, poly_yrad, 1)
    xr, yr, zr = sph2cart_array(points_xrad, points_yrad, 1)
    nc = np.size(xc,0)
    nr = np.size(xr,0)
    
    # Plate center
    pmx = np.mean(xc);
    pmy = np.mean(yc);
    pmz = np.mean(zc);
    angle = np.arctan2(np.linalg.norm(np.cross(np.array([pmx, pmy, pmz]), np.array([0., 0., 1.]))), 
                       np.dot(np.array([0., 0., 1.]),np.array([pmx, pmy, pmz]))
                       ) 
    
    # Rotation axis
    rot = np.cross(np.array([pmx, pmy, pmz]),np.array([0, 0, 1]))
    rot = rot/np.linalg.norm(rot)

    # Rotation matrix
    rot_mtx = finite_rotation_matrix(rot[0],rot[1],rot[2],angle)

    
    
    bou_pole = np.zeros([nc,3])
    for i2 in range(nc):
        aux1 = np.dot(rot_mtx, np.array([ xc[i2], yc[i2], zc[i2] ]))
        bou_pole[i2,:] = aux1
    
    ndrg_pole = np.zeros([nr,3])
    for i2 in range(nr):
        aux1 = np.dot(rot_mtx, np.array([ xr[i2], yr[i2], zr[i2] ]))
        ndrg_pole[i2,:] = aux1
        
    
    bou_list = [[x,y] for x,y in zip(bou_pole[:,0], bou_pole[:,1])]
    pnt_list = [[x,y] for x,y in zip(ndrg_pole[:,0], ndrg_pole[:,1])]
    
    p = path.Path(bou_list)
    inpo = p.contains_points(pnt_list,radius=0.05)
    
    boolz_list = ndrg_pole[:,2] < 0  #filter values with same xy, but negative z
    inpo[boolz_list] = False
    
    return inpo



def gridpoints_in_polygon(polygon_xdeg, polygon_ydeg, grid_xdeg, grid_ydeg):
    
    regrid_x = grid_xdeg.flatten()
    regrid_y = grid_ydeg.flatten()
    
    inpo = points_in_polygon(polygon_xdeg, polygon_ydeg, regrid_x, regrid_y)
    inpo_grid = np.reshape(inpo, (np.size(grid_xdeg,0), np.size(grid_xdeg,1)))
    
    return inpo_grid



def filter_grid(filter_grid, points_grid):
    
    points_filt = np.full(filter_grid.shape, np.nan)
    points_filt[filter_grid] = points_grid[filter_grid]
    
    return points_filt