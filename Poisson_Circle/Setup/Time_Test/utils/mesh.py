from mshr import *
from dolfin import *

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

import png
from PIL import Image, ImageFilter


# Generates mesh for a polygon with vertex count in specified interval,
# saves mesh for solver, and saves image file of specified resolution.
def gen_mesh(resolution, mesh_resolution, mesh_directory, ID, coarse_resolution):

    # Construct domain and mesh using Dolfin/mshr
    #
    # Note: mshr uses CGAL and TetGen to construct
    # a Delaunay triangulation of the domain.
    #
    center = Point(0.5,0.5)
    domain = Circle(center,0.45)
    mesh = generate_mesh(domain,mesh_resolution)

    # Generate coarse mesh for comparisons
    coarse_mesh = generate_mesh(domain,coarse_resolution)


    # Define discretization of unit square
    #[x_min, x_max] = [-1.0, 1.0]
    #[y_min, y_max] = [-1.0, 1.0]
    [x_min, x_max] = [0.0, 1.0]
    [y_min, y_max] = [0.0, 1.0]
    x_step = (x_max - x_min)/resolution
    y_step = (y_max - y_min)/resolution
    
    mesh_array = np.zeros((resolution,resolution),dtype='uint8')
    alpha_array = np.zeros((resolution,resolution),dtype='uint8')

    pixel_count = 0
    # Determine which points lie within the domain
    for j in range(0,resolution):
        for i in range(0,resolution):
            # Starts at upper left corner of square
            x_i = x_min + (i+0.5)*x_step
            y_j = y_max - (j+0.5)*y_step
            pt = Point(x_i,y_j)
            cell_id = mesh.bounding_box_tree().compute_first_entity_collision(pt)
            #if mesh.bounding_box_tree().collides(pt):
            if cell_id < mesh.num_cells():
                # Point lies within domain
                mesh_array[j,i] = 0
                alpha_array[j,i] = 255
                pixel_count += 1
            else:
                # Point lies outside domain
                mesh_array[j,i] = 255
                alpha_array[j,i] = 0




    # Define hi-res discretization of unit square
    new_resolution = 2*resolution
    new_x_step = (x_max - x_min)/new_resolution
    new_y_step = (y_max - y_min)/new_resolution
    
    new_mesh_array = np.zeros((new_resolution,new_resolution),dtype='uint8')
    new_alpha_array = np.zeros((new_resolution,new_resolution),dtype='uint8')

    new_pixel_count = 0
    # Determine which points lie within the domain
    for j in range(0,new_resolution):
        for i in range(0,new_resolution):
            # Starts at upper left corner of square
            x_i = x_min + (i+0.5)*new_x_step
            y_j = y_max - (j+0.5)*new_y_step
            pt = Point(x_i,y_j)
            cell_id = mesh.bounding_box_tree().compute_first_entity_collision(pt)
            #if mesh.bounding_box_tree().collides(pt):
            if cell_id < mesh.num_cells():
                # Point lies within domain
                new_mesh_array[j,i] = 0
                new_alpha_array[j,i] = 255
                new_pixel_count += 1
            else:
                # Point lies outside domain
                new_mesh_array[j,i] = 255
                new_alpha_array[j,i] = 0



    # Normalize input data array
    def normalize_alpha(A):
        out_of_domain = (A == 0)
        inside_domain = (A == 255)
        A[out_of_domain] = 0
        A[inside_domain] = 1
        return A


    # Convert Arrays to Image
    img = Image.fromarray(mesh_array, mode='L')
    mask = Image.fromarray(alpha_array, mode='L')
    img.convert('LA')
    img.putalpha(mask)

    # Save Mesh Image
    #image_filename = mesh_directory + 'mesh_' + str(ID) + '.png'
    #img.save(image_filename)

    # Save Mesh Array
    vals, dom = img.split()
    bdry_image = img.filter(ImageFilter.FIND_EDGES)
    vals, bdry = bdry_image.split()        

    domain_array = np.array(dom, dtype='uint8')
    boundary_array = np.array(bdry, dtype='uint8')

    domain_array = normalize_alpha(domain_array)
    boundary_array = normalize_alpha(boundary_array)
    domain_boundary = domain_array + boundary_array

    mesh_array_file = mesh_directory + 'mesh_' + str(ID) + '.npy'
    np.save(mesh_array_file,domain_boundary)

    # Save Mesh Data
    mesh_filename = mesh_directory + 'mesh_' + str(ID) + '.xml'
    File(mesh_filename) << mesh

    # Save Coarse Mesh Data
    coarse_mesh_filename = mesh_directory + 'coarse_mesh_' + str(ID) + '.xml'
    File(coarse_mesh_filename) << coarse_mesh


    # Convert Hi-Res Arrays to Image
    new_img = Image.fromarray(new_mesh_array, mode='L')
    new_mask = Image.fromarray(new_alpha_array, mode='L')
    new_img.convert('LA')
    new_img.putalpha(new_mask)

    # Save Mesh Image
    #image_filename = mesh_directory + 'mesh_' + str(ID) + '.png'
    #img.save(image_filename)

    # Save Mesh Array
    vals, dom = new_img.split()
    bdry_image = new_img.filter(ImageFilter.FIND_EDGES)
    vals, bdry = bdry_image.split()        

    domain_array = np.array(dom, dtype='uint8')
    boundary_array = np.array(bdry, dtype='uint8')

    domain_array = normalize_alpha(domain_array)
    boundary_array = normalize_alpha(boundary_array)
    domain_boundary = domain_array + boundary_array

    mesh_array_file = mesh_directory + 'hires_mesh_' + str(ID) + '.npy'
    np.save(mesh_array_file,domain_boundary)

    # Save Mesh Data (same as lo-res)
    #mesh_filename = mesh_directory + 'hires_mesh_' + str(ID) + '.xml'
    #File(mesh_filename) << new_mesh

