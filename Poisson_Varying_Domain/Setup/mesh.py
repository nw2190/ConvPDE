from mshr import *
from dolfin import *

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

import png
from PIL import Image, ImageFilter


# Generates mesh for a polygon with vertex count in specified interval,
# saves mesh for solver, and saves image file of specified resolution.
def gen_mesh(resolution, vertex_min, vertex_max, mesh_resolution, mesh_directory, ID):

    # Define the Number of Vertices
    N = np.random.randint(vertex_min,vertex_max)

    # Define maximum ratios to maintain regularity
    max_radius_ratio = 4.0
    #max_angle_ratio = 8.0
    max_angle_ratio = 2.0

    min_radius = 1.0/max_radius_ratio
    min_angle = 1.0/max_angle_ratio

    # Define minimum number of pixels domain must contain
    #min_pixels = resolution*resolution//4
    min_pixels = int(resolution*resolution//3.75)
    
    # Define list of radii (normalized to have max radius of 1)
    radii = np.random.uniform(min_radius,1.0,N)
    normalized_radii = np.zeros((N,))

    # Smooth neighboring vertices for increased regularity
    smoothing = 1
    weight = 0.3333
    for k in range(0,N):
        sum = 0.0
        for j in range(-smoothing,smoothing+1):
            if j == 0:
                sum = sum + radii[np.mod(k+j,N)]
            else:
                sum = sum + weight*radii[np.mod(k+j,N)]
            normalized_radii[k] = sum/(2*smoothing*weight+1)

    max_radius = np.max(normalized_radii)
    normalized_radii = 1.0/max_radius*normalized_radii

    
    # Define Increments in Vertex Angles
    angles = np.random.uniform(min_angle,1.0,N)
    full_sum = 0.0
    for k in range(0,N):
        full_sum = full_sum + angles[k]

    percent_angle = np.cumsum(1.0/full_sum*angles)


    # Construct list of vertices as Dolfin Points
    vertices = list()
    for k in range(0,N):
        theta = 2*np.pi*percent_angle[k]
        r = normalized_radii[k]
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        x = 0.5*(x+1.0)
        y = 0.5*(y+1.0)
        pt = Point(x,y)
        vertices.append(pt)


    #
    # Construct domain and mesh using Dolfin/mshr
    #
    # Note: mshr uses CGAL and TetGen to construct
    # a Delaunay triangulation of the domain.
    #
    domain = Polygon(vertices)
    mesh = generate_mesh(domain,mesh_resolution)


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
            p = Point(x_i,y_j)
            #if mesh.bounding_box_tree().collides(p):

            #cell_id = mesh.bounding_box_tree().compute_first_entity_collision(pt)
            #if cell_id < mesh.num_cells():

            collision = mesh.bounding_box_tree().compute_first_collision(p)
            if not (collision == 4294967295):
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
            p = Point(x_i,y_j)
            #if mesh.bounding_box_tree().collides(p):

            #cell_id = mesh.bounding_box_tree().compute_first_entity_collision(pt)
            #if cell_id < mesh.num_cells():

            collision = mesh.bounding_box_tree().compute_first_collision(p)
            if not (collision == 4294967295):
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


    if pixel_count < min_pixels:
        #print('Domain is too small; regenerating...')
        gen_mesh(resolution, vertex_min, vertex_max, mesh_resolution, mesh_directory, ID)

    else:
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


def gen_mesh_batch(resolution, vertex_min, vertex_max, mesh_resolution, mesh_dir, data_count, current_data):

    # Reset random seed from parent process
    np.random.seed(seed=current_data)
    
    for n in range(current_data,current_data + data_count):
        gen_mesh(resolution, vertex_min, vertex_max, mesh_resolution, mesh_dir, n)
